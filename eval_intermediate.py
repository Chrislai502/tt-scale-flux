import os
import json
from datetime import datetime
from PIL import Image
import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLPipeline
from tqdm.auto import tqdm
import tempfile
from diffusers.utils import export_to_video
from dotenv import load_dotenv
import subprocess
import time
# from verifiers.image_reward import ImageRewardVerifier

subprocess.run(["bash", "setup_env.sh"])
load_dotenv()

HF_HOME = os.getenv("HF_HOME")
print(f"Using HF_HOME: {HF_HOME}")

from utils import (
    generate_neighbors,
    prompt_to_filename,
    get_noises,
    TORCH_DTYPE_MAP,
    get_latent_prep_fn,
    parse_cli_args,
    serialize_artifacts,
    MODEL_NAME_MAP,
    prepare_video_frames,
)
from verifiers import SUPPORTED_VERIFIERS

# Non-configurable constants
TOPK = 1  # Always selecting the top-1 noise for the next round
MAX_SEED = np.iinfo(np.int32).max  # To generate random seeds


def sample(
    noises: dict[int, torch.Tensor],
    prompt: str,
    search_round: int,
    pipe: DiffusionPipeline,
    verifier,
    topk: int,
    root_dir: str,
    config: dict,
    extract_intermediate_scores: bool = True
) -> dict:
    """
    For a given prompt, generate images using all provided noises in batches,
    score them with the verifier, and select the top-K noise.
    The images and JSON artifacts are serialized via `serialize_artifacts`.
    """
    use_low_gpu_vram = config.get("use_low_gpu_vram", False)
    batch_size_for_img_gen = config.get("batch_size_for_img_gen", 1)
    verifier_args = config.get("verifier_args")
    choice_of_metric = verifier_args.get("choice_of_metric", None)
    verifier_to_use = verifier_args.get("name", "gemini")
    search_args = config.get("search_args", None)
    export_args = config.get("export_args", {})  # Define export_args at function start


    images_for_prompt = []
    noises_used = []
    seeds_used = []
    images_info = []  # Will collect (seed, noise, image, filename) tuples for serialization.
    prompt_filename = prompt_to_filename(prompt)

    # Convert the noises dictionary into a list of (seed, noise) tuples.
    noise_items = list(noises.items())

    # Process the noises in batches.
    # TODO: find better way
    extension_to_use = "png"
    if "LTX" in pipe.__class__.__name__:
        extension_to_use = "mp4"
    elif "Wan" in pipe.__class__.__name__:
        extension_to_use = "mp4"
        
    score_output= {}
        
    # Process the noises in batches to maximize GPU efficiency
    for i in range(0, len(noise_items), batch_size_for_img_gen):
        
        # Extract a batch of noise items
        batch = noise_items[i : i + batch_size_for_img_gen]
        seeds_batch, noises_batch = zip(*batch) # Separate seeds and noises
        
        # Create filenames for each output in the batch
        # Format: prompt_i@search_round_s@seed.extension
        filenames_batch = [
            os.path.join(root_dir, f"{prompt_filename}_i@{search_round}_s@{seed}.{extension_to_use}")
            for seed in seeds_batch
        ]
        seed = seeds_batch[0]
        save_intermediate_images_path = os.path.join(root_dir, "latent_images", str(seed))

        # Manage GPU memory - move model to GPU if using low VRAM mode
        # (except for the "gemini" verifier which may handle this differently)
        if use_low_gpu_vram and verifier_to_use != "gemini":
            pipe = pipe.to("cuda:0") # Move pipeline to GPU
        print(f"Generating images for batch with seeds: {list(seeds_batch)}.")

        # Create a list of identical prompts (one for each noise in the batch)
        batched_prompts = [prompt] * len(noises_batch)
        batched_latents = torch.stack(noises_batch).squeeze(dim=1)

        # Generate images using the diffusion pipeline
        # Pass the batched prompts, noise tensors, and additional arguments from config
        print("Generating images...")
        start_time = time.time()
        
        # --- set up timing callback to record wall‐clock time at each diffusion step ---
        step_times: dict[int, float] = {}
        def timing_callback(pipeline, step: int, timestep, callback_kwargs):
            # timestep is a torch scalar or int; normalize to Python int
            t = int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
            step_times[t] = time.time() - start_time
            return {}  # do not modify anything
        
        batch_result = pipe(prompt=batched_prompts, latents=batched_latents, callback_on_step_end=timing_callback, save_latent_images=save_intermediate_images_path, **config["pipeline_call_args"])
        end_time = time.time()
        print(f"Generated images in {end_time - start_time:.2f} seconds.")
        
        # Extract the intermediate scores using the verifier
        if extract_intermediate_scores:
            # score_output[i] = {}
            image_files = [(f, float(f.split('.png')[0].split('_')[-1])) for f in os.listdir(save_intermediate_images_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            # sort the iamge files by name
            image_files.sort(key=lambda x: x[1])
            
            # Make a list of the images in prepare for the verifier
            # intermediate_images_to_evaluate = []
            # timesteps = []
            verifier_batch_size = 3
            
            # Perform verification in batches loading images on the fly
            print("Performing verification on intermediate images...")
            total_start_time = time.time()
            total_images = len(image_files)
            processed_images = 0

            while processed_images < total_images:
                # Get current batch of image files
                batch_files = image_files[processed_images:processed_images + verifier_batch_size]
                batch_size = len(batch_files)
                
                # Load images and timesteps for current batch on the fly
                batch_images = []
                batch_timesteps = []
                
                for f, timestep in batch_files:
                    timestep = int(timestep)
                    image = Image.open(os.path.join(save_intermediate_images_path, f))
                    batch_images.append(image)
                    batch_timesteps.append(timestep)
                
                # Process the batch
                start_time = time.time()
                if batch_images and isinstance(batch_images[0], Image.Image):
                    # import ipdb; ipdb.set_trace()
                    verifier_inputs = verifier.prepare_inputs(images=batch_images, prompts=[prompt] * batch_size)
                    batch_outputs = verifier.score(inputs=verifier_inputs)
                else:
                    raise NotImplementedError
                end_time = time.time()
                
                print(f"Batch verification took {end_time - start_time} seconds for {batch_size} images.")
                print(f"Batch verification took {(end_time - start_time)/batch_size} seconds per image.")
                
                # Post-process the outputs to store them
                for t_, verifier_scores_output in zip(batch_timesteps, batch_outputs):
                    # Initialize nested dictionaries if they don't exist
                    if str(seed) not in score_output:
                        score_output[str(seed)] = {
                            "round": search_round,
                            "intermediates": {}
                        }
                    
                    score_output[str(seed)]["intermediates"][str(t_)] = {
                        **verifier_scores_output,
                        "time_so_far": step_times.get(t_, None)
                    }
                
                # Clean up memory by closing images after processing
                for img in batch_images:
                    img.close()
                
                # Update the number of processed images
                processed_images += batch_size

            total_end_time = time.time()
            print(f"Total verification took {total_end_time - total_start_time} seconds for {total_images} images.")
            print(f"Average verification took {(total_end_time - total_start_time)/total_images} seconds per image.")
                        
            # for f, timestep in image_files:
            #     timestep = int(timestep)
            #     image = Image.open(os.path.join(save_intermediate_images_path, f))
            #     intermediate_images_to_evaluate.append(image)
            #     timesteps.append(timestep)
            
            # # Perform the verification
            # print("Performing verification on intermediate images...")
            # start_time = time.time()
            # if isinstance(intermediate_images_to_evaluate[0], Image.Image):
            #     verifier_inputs = verifier.prepare_inputs(images=intermediate_images_to_evaluate, prompts=[prompt] * len(intermediate_images_to_evaluate))
            #     outputs = verifier.score(inputs=verifier_inputs)
            # else:
            #     raise NotImplementedError
            # end_time = time.time()
            # print(f"Verification took {end_time - start_time} seconds for {len(intermediate_images_to_evaluate)} images.")
            # print(f"Verification took {(end_time - start_time)/len(intermediate_images_to_evaluate)} seconds per image.")
            
            # # Post-process the outputs to store them
            # for t_, verifier_scores_output in zip(timesteps, outputs):
            #     score_output[str(seed)][str(t_)] = verifier_scores_output
                
        # Extract the generated images or video frames based on output type
        if hasattr(batch_result, "images"):
            batch_images = batch_result.images
        elif hasattr(batch_result, "frames"):
            batch_images = [vid for vid in batch_result.frames]

        print(f"INFORMATION: batch_images type: {type(batch_images[0])}")
        
        # If using low VRAM mode, move pipeline back to CPU to free up GPU memory
        if use_low_gpu_vram and verifier_to_use != "gemini":
            pipe = pipe.to("cpu")

        # Collect all the generated outputs and their corresponding metadata
        for seed, noise, image, filename in zip(seeds_batch, noises_batch, batch_images, filenames_batch):
            images_for_prompt.append(image)
            noises_used.append(noise)
            seeds_used.append(seed)
            images_info.append((seed, noise, image, filename))


    # Save the score outputs as a JSON file
    if extract_intermediate_scores:
        score_output_path = os.path.join(root_dir, "score_output.json")
        if os.path.exists(score_output_path):
            with open(score_output_path, 'r') as file:
                existing_score_output_data = json.load(file)

            existing_score_output_data.update(score_output)

            with open(score_output_path, "w") as f:
                json.dump(existing_score_output_data, f)
                print(f"Saved round {search_round} scores")
        else:
            with open(score_output_path, "w") as f:
                json.dump(score_output, f)
                print(f"Saved round {search_round} scores")
        

    # Prepare verifier inputs and perform inference.
    start_time = time.time()
    if isinstance(images_for_prompt[0], Image.Image):
        verifier_inputs = verifier.prepare_inputs(images=images_for_prompt, prompts=[prompt] * len(images_for_prompt))
    else:
        export_args = config.get("export_args", None) or {}
        if export_args:
            fps = export_args.get("fps", 24)
        else:
            fps = 24
        temp_vid_paths = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, vid in enumerate(images_for_prompt):
                vid_path = os.path.join(tmpdir, f"{idx}.mp4")
                export_to_video(vid, vid_path, fps=fps)
                temp_vid_paths.append(vid_path)

            verifier_inputs = []
            for vid_path in temp_vid_paths:
                frames = prepare_video_frames(vid_path)
                verifier_inputs.append(verifier.prepare_inputs(images=frames, prompts=[prompt] * len(frames)))

    print("Scoring with the verifier.")
    print(f"Verifier inputs: {verifier_inputs.keys()}")
    print(f"Verifier inputs length: {len(verifier_inputs['images'])}")
    outputs = verifier.score(inputs=verifier_inputs)
    for o in outputs:
        assert choice_of_metric in o, o.keys()

    assert (
        len(outputs) == len(images_for_prompt)
    ), f"Expected len(outputs) to be same as len(images_for_prompt) but got {len(outputs)=} & {len(images_for_prompt)=}"

    results = []
    
    end_time = time.time()
    print(f"Time taken for Verifier: {end_time - start_time}")
    
    print(outputs)
    for json_dict, seed_val, noise in zip(outputs, seeds_used, noises_used):
        # Merge verifier outputs with noise info.
        merged = {**json_dict, "noise": noise, "seed": seed_val}
        results.append(merged)

    def f(x):
        # If the verifier output is a dict, assume it contains a "score" key.
        if isinstance(x[choice_of_metric], dict):
            return x[choice_of_metric]["score"]
        return x[choice_of_metric]

    sorted_list = sorted(results, key=lambda x: f(x), reverse=True)
    topk_scores = sorted_list[:topk]

    # Print debug information.
    for ts in topk_scores:
        print(f"Prompt='{prompt}' | Best seed={ts['seed']} | Score={ts[choice_of_metric]}")

    best_img_path = os.path.join(
        root_dir, f"{prompt_filename}_i@{search_round}_s@{topk_scores[0]['seed']}.{extension_to_use}"
    )
    datapoint = {
        "prompt": prompt,
        "search_round": search_round,
        "num_noises": len(noises),
        "best_noise_seed": topk_scores[0]["seed"],
        "best_noise": topk_scores[0]["noise"],
        "best_score": topk_scores[0][choice_of_metric],
        "choice_of_metric": choice_of_metric,
        "best_img_path": best_img_path,
    }

    # Check if the neighbors have any improvements (zero-order only).
    search_method = search_args.get("search_method", "random") if search_args else "random"
    if search_args and search_method == "zero-order":
        first_score = f(results[0])
        neighbors_with_better_score = any(f(item) > first_score for item in results[1:])
        datapoint["neighbors_improvement"] = neighbors_with_better_score

    # Serialize.
    if search_method == "zero-order":
        if datapoint["neighbors_improvement"]:
            serialize_artifacts(images_info, prompt, search_round, root_dir, datapoint, **export_args)
        else:
            print("Skipping serialization as there was no improvement in this round.")
    elif search_method == "random":
        serialize_artifacts(images_info, prompt, search_round, root_dir, datapoint, **export_args)

    return datapoint


@torch.no_grad()
def main():
    # === Load configuration and CLI arguments ===
    args = parse_cli_args()
    with open(args.pipeline_config_path, "r") as f:
        config = json.load(f)
    config.update(vars(args))

    search_args = config["search_args"]
    search_rounds = search_args["search_rounds"]
    search_method = search_args.get("search_method", "random")
    num_prompts = config["num_prompts"]

    # === Create output directory ===
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_name = config.pop("pretrained_model_name_or_path")
    verifier_name = config["verifier_args"]["name"]
    choice_of_metric = config["verifier_args"]["choice_of_metric"]
    output_dir = os.path.join(
        "output",
        MODEL_NAME_MAP[pipeline_name],
        verifier_name,
        choice_of_metric,
        current_datetime,
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Artifacts will be saved to: {output_dir}")
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # === Load prompts ===
    if args.prompt is None:
        with open("prompts_open_image_pref_v1.txt", "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        if num_prompts != "all":
            prompts = prompts[:num_prompts]
    else:
        prompts = [args.prompt]
    print(f"Using {len(prompts)} prompt(s).")

    # === Set up the image-generation pipeline ===
    print("Loading pipeline...")
    torch_dtype = TORCH_DTYPE_MAP[config.pop("torch_dtype")]
    fp_kwargs = {"pretrained_model_name_or_path": pipeline_name, "torch_dtype": torch_dtype}
    if "Wan" in pipeline_name:
        # As per recommendations from https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan.
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(pipeline_name, subfolder="vae", torch_dtype=torch.float32, cache_dir=HF_HOME)
        fp_kwargs.update({"vae": vae})
    pipe = DiffusionPipeline.from_pretrained(**fp_kwargs, cache_dir=HF_HOME)
    if not config.get("use_low_gpu_vram", False):
        pipe = pipe.to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    print("The type of pipe used")
    print(pipe)
    print(pipe.scheduler)
    # print(pipe.scheduler.timesteps)
    # pipe.scheduler.set_timesteps(100)
    print("Pipeline loaded.")

    # === Load verifier model ===
    print("Loading verifier...")
    verifier_args = config["verifier_args"]
    verifier_cls = SUPPORTED_VERIFIERS.get(verifier_args["name"])
    if verifier_cls is None:
        raise ValueError("Verifier class evaluated to be `None`. Make sure the dependencies are installed properly.")

    verifier = verifier_cls(**verifier_args)
    print("Verifier loaded.")

    # === Main loop: For each prompt and each search round ===
    pipeline_call_args = config["pipeline_call_args"].copy()
    for prompt in tqdm(prompts, desc="Processing prompts"):
        search_round = 1

        # For zero-order search, we store the best datapoint per round.
        best_datapoint_per_round = {}

        while search_round <= search_rounds:
            # Determine the number of noise samples.
            if search_method == "zero-order":
                num_noises_to_sample = 1
            else:
                num_noises_to_sample = 2**search_round

            print(f"\n=== Prompt: {prompt} | Round: {search_round} ===")

            # --- Generate noise pool ---
            should_regenate_noise = True
            previous_round = search_round - 1
            if previous_round in best_datapoint_per_round:
                was_improvement = best_datapoint_per_round[previous_round]["neighbors_improvement"]
                if was_improvement:
                    should_regenate_noise = False

            # For subsequent rounds in zero-order: use best noise from previous round.
            # This happens ONLY if there was an improvement with the neighbors in the
            # previous round, otherwise round is progressed with newly sampled noise.
            if should_regenate_noise:
                # Standard noise sampling.
                if search_method == "zero-order" and search_round != 1:
                    print("Regenerating base noise because the previous round was rejected.")
                noises = get_noises(
                    max_seed=MAX_SEED,
                    num_samples=num_noises_to_sample,
                    dtype=torch_dtype,
                    fn=get_latent_prep_fn(pipeline_name),
                    **pipeline_call_args,
                )
            else:
                if best_datapoint_per_round[previous_round]:
                    if best_datapoint_per_round[previous_round]["neighbors_improvement"]:
                        print("Using the best noise from the previous round.")
                        prev_dp = best_datapoint_per_round[previous_round]
                        noises = {int(prev_dp["best_noise_seed"]): prev_dp["best_noise"]}

            if search_method == "zero-order":
                # Process the noise to generate neighbors.
                base_seed, base_noise = next(iter(noises.items()))
                neighbors = generate_neighbors(
                    base_noise, threshold=search_args["threshold"], num_neighbors=search_args["num_neighbors"]
                ).squeeze(0)
                # Concatenate the base noise with its neighbors.
                neighbors_and_noise = torch.cat([base_noise, neighbors], dim=0)
                new_noises = {}
                for i, noise_tensor in enumerate(neighbors_and_noise):
                    new_noises[base_seed + i] = noise_tensor.unsqueeze(0)
                noises = new_noises

            print(f"Number of noise samples for prompt '{prompt}': {len(noises)}")

            # --- Sampling, verifying, and saving artifacts ---
            datapoint = sample(
                noises=noises,
                prompt=prompt,
                search_round=search_round,
                pipe=pipe,
                verifier=verifier,
                topk=TOPK,
                root_dir=output_dir,
                config=config,
            )

            if search_method == "zero-order":
                # Update the best datapoint for zero-order.
                if datapoint["neighbors_improvement"]:
                    best_datapoint_per_round[search_round] = datapoint

            search_round += 1


if __name__ == "__main__":
    main()

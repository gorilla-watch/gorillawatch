#!/bin/bash

# Usage: bash ./scripts/run-in-docker.sh [OPTIONS] [COMMAND]
# ---------------------
# Options:
#   -g   comma-separated GPU ids to expose, or "all" (default: none)
#   -i   docker image to use (default: gorillawatch:1.2)
# ---------------------
# Open a shell without GPUs:            bash ./scripts/run-in-docker.sh bash
# Open a shell on GPUs 0 and 1:         bash ./scripts/run-in-docker.sh -g 0,1 bash
# Train on GPU 0:                       bash ./scripts/run-in-docker.sh -g 0 ./scripts/train.sh
# Evaluate with a custom image:         bash ./scripts/run-in-docker.sh -g 0 -i my/image:tag ./scripts/eval_hf_fine_tuned.sh
# ---------------------
# The project is mounted at /workspaces/gorillawatch and "pip install -e ." runs before COMMAND.
# The HuggingFace cache is shared with the host, so datasets and pretrained backbones are downloaded only once.
# W&B credentials are read from $WANDB_API_KEY or from ~/.netrc (run "wandb login" once on the host).

set -e

# Default values
image="gorillawatch:1.2"
command="bash"
gpus="none"

# Check if the current directory is named "scripts"
if [ "$(basename "$PWD")" == "scripts" ]; then
    echo "Error: This script should be called from the root of the project."
    echo "Example: bash ./scripts/run-in-docker.sh"
    exit 1
fi

# Function to parse the command line arguments
parse_arguments() {
    local in_command=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
        -g)
            shift
            gpus="$1"
            ;;
        -i)
            shift
            image="$1"
            ;;
        *)
            if [ "$in_command" = false ]; then
                command="$1"
            else
                command="${command} $1"
            fi
            in_command=true
            ;;
        esac
        shift
    done
    command="pip install -e . && ${command}"
}

# Call the function to parse arguments
parse_arguments "$@"

echo "image: $image"
echo "command: $command"
echo "gpus: $gpus"

# Look for WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
    if [ -f "$HOME/.netrc" ]; then
        export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; getline; print $2}' "$HOME/.netrc")
    fi
    if [ -z "$WANDB_API_KEY" ]; then
        echo "WANDB_API_KEY not found - runs will not be logged to W&B"
    else
        echo "WANDB_API_KEY found in ~/.netrc"
    fi
else
    echo "WANDB_API_KEY found in environment"
fi

# Build the image on first use
if [ -z "$(docker images -q "$image")" ]; then
    echo "Image $image not found locally, building it from ./Dockerfile ..."
    DOCKER_BUILDKIT=1 docker build -t "$image" .
fi

# Share the host HuggingFace cache so datasets and backbone weights are downloaded only once.
# Created here so the directory is not owned by root afterwards.
mkdir -p "$HOME/.cache/huggingface"

# NOTE: --ipc=host gives full RAM and CPU access, use -m XXXG --cpus XX to restrict it instead.
# Add mounts for data that lives outside the project, e.g. -v /scratch/username/data:/workspaces/data
# Add -p 5678:5678 to expose a port for remote debugging (blocks the port for other docker users on the host).
docker_args=(
    --rm -it --ipc=host --network=host
    -v "${PWD}:/workspaces/gorillawatch"
    -w /workspaces/gorillawatch
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface"
    # root, so that "pip install -e ." can write into the conda environment of the image
    --user 0:0
    --env WANDB_API_KEY
    --name "gorillawatch-$(id -un)-${gpus//,/}"
)

# Mount ~/.netrc as a fallback for tools that read the W&B credentials from disk
if [ -f "$HOME/.netrc" ]; then
    docker_args+=(-v "${HOME}/.netrc:/root/.netrc:ro")
fi

if [ "$gpus" == "all" ]; then
    docker_args+=(--gpus all)
elif [ "$gpus" != "none" ]; then
    docker_args+=(--gpus "device=${gpus}")
fi

docker run "${docker_args[@]}" "$image" /bin/bash -c "${command}"

echo 'Done'

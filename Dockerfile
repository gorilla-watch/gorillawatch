# syntax=docker/dockerfile:1.5.2

# This Dockerfile produces a container with all dependencies installed into an environment called "research"
# Additionally, the container has a full-fledged micromamba installation, which is a faster drop-in replacement for conda
# When inside the container, you can install additional dependencies with `mamba install <package>`, e.g. `mamba install scipy`
# The actual installation is done by micromamba, we have simply provided an alias to link the mamba command to micromamba

# The syntax line above is crucial to enable variable expansion for type=cache=mount commands

# We can use the OS_PREFIX build arg to choose between ubi8 and ubuntu as base image (for amd64 processor architecture)
ARG OS_SELECTOR=ubuntu

# Load micromamba container to copy from later
FROM --platform=$TARGETPLATFORM mambaorg/micromamba:1.4.2 as micromamba


####################################################
################ BASE IMAGES #######################
####################################################

# -----------------
# base image for amd64
# -----------------
FROM --platform=linux/amd64 nvidia/cuda:12.2.2-cudnn8-devel-ubi8 as amd64ubi8
# Install compiler for .compile() with PyTorch 2.0 and nano for devcontainers
RUN yum install -y git gcc gcc-c++ nano && yum clean all
# Copy lockfile to container
COPY environment.yml /locks/environment.yml

# -----------------
# devcontainer base image for amd64 using Ubuntu
# SLURM + pyxis has a bug on our cluster, where the automatic activation of the conda environment fails if the base image is ubuntu
# But Ubuntu works better for devcontainers than ubi8
# So we use Ubuntu for devcontainers and ubi8 for actual deployment on the cluster
# -----------------
FROM --platform=linux/amd64 nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 as amd64ubuntu
# Install compiler for .compile() with PyTorch 2.0 and nano for devcontainers
RUN apt-get update && apt-get install -y git gcc g++ nano openssh-client && apt-get clean
# Copy lockfile to container
COPY environment.yml /locks/environment.yml





####################################################
################ FINAL IMAGE #######################
####################################################

# -----------------
# Final build image - we choose the correct base image based on the target architecture and OS
# -----------------
ARG TARGETARCH
FROM ${TARGETARCH}${OS_SELECTOR} as final
# From https://github.com/mamba-org/micromamba-docker#adding-micromamba-to-an-existing-docker-image
# The commands below add micromamba to an existing image to give the capability to ad-hoc install new dependencies

####################################################
######### Adding micromamba starts here ############
####################################################
USER root

# if your image defaults to a non-root user, then you may want to make the
# next 3 ARG commands match the values in your image. You can get the values
# by running: docker run --rm -it my/image id -a
ARG MAMBA_USER=mamba
ARG MAMBA_USER_ID=1000
ARG MAMBA_USER_GID=1000
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh

# Install system dependencies
RUN apt-get update && apt-get install -y \
    make \
    bash-completion \
    postgresql-client \
    libcurl4 \
    htop \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "source /usr/share/bash-completion/bash_completion" >> /root/.bashrc

# Install nvm for our frontend
ENV NVM_DIR=/root/.nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash 
RUN . $NVM_DIR/nvm.sh && nvm install --lts

USER $MAMBA_USER

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["/bin/bash"]

####################################################
######### Adding micromamba stops here #############
####################################################

# Switch to root user to grant necessary permissions and setup
USER root
RUN chown $MAMBA_USER:$MAMBA_USER /usr/bin/gcc && \
    chown -R $MAMBA_USER:$MAMBA_USER /locks/ && \
    echo "alias mamba=micromamba" >> /usr/local/bin/_activate_current_env.sh && \
    mkdir -p /home/mamba/.cache && chmod -R 777 /home/mamba/.cache/

# Switch back to micromamba user
USER $MAMBA_USER
ARG TARGETPLATFORM
# Install dependencies from lockfile into environment, cache packages in /opt/conda/pkgs
RUN --mount=type=cache,target=$MAMBA_ROOT_PREFIX/pkgs,id=conda-$TARGETPLATFORM,uid=$MAMBA_USER_ID,gid=$MAMBA_USER_GID \
    micromamba create --name research --yes --file /locks/environment.yml

# Configure micromamba
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba config prepend channels conda-forge --env && \
    micromamba config set show_banner false --env

# Fix opencv problems and install additional pip dependencies
RUN micromamba run -n research pip install opencv-python-headless --force --no-deps --no-cache-dir

# Set environment variable for default conda environment
ENV ENV_NAME=research

RUN pip install flash-attn==2.8.2 --no-build-isolation --use-pep517

# Install drop_layer_norm for InternVideo
# WORKDIR /home/${MAMBA_USER}
# RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    # cd flash-attention/csrc/layer_norm && \
    # pip install .

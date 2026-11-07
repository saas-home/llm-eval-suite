# ============================================================================
# llm-eval-suite: evaluation image with compilers and test frameworks for
# Python, TypeScript, Go, Java, Rust and C#.
#
# Base: Ubuntu 26.04 LTS (x86_64 and aarch64)
#   Python 3.14 + mypy    Ubuntu package
#   Java: OpenJDK 25      Ubuntu package (current Java LTS)
#   Node.js 26            NodeSource (current Node.js LTS)
#   TypeScript 7          npm, pinned in scripts/setup_polyglot_toolchains.sh
#   Go                    go.dev release, pinned below with SHA-256
#   Rust                  rustup, pinned toolchain below
#   .NET 10 SDK           Microsoft installer (current .NET LTS)
#
# To upgrade a toolchain, change its ARG, rebuild, and run
# scripts/verify_polyglot_packs.py in the new image.
# ============================================================================

FROM ubuntu:26.04

ARG GO_VERSION=1.27.3
ARG GO_SHA256_AMD64=65efbc80fdc1fa2cd2631cad5b481777a68e8725f34e89f50fc6ac714e379b40
ARG GO_SHA256_ARM64=a06e99309dbcbdca1a8ccd2282c3ee146e55930886e26b4f1193269587165f49
ARG RUSTUP_VERSION=1.29.3
ARG RUSTUP_SHA256_AMD64=b1c8cb14e30889e7e2999e6b69e62d9e9b9ffc0f4df8180564d306d9680e6c55
ARG RUSTUP_SHA256_ARM64=9f608c142e0e38a78957d0661ba1dfaf0b6fb37184cfaa4b9a0aa57af6184282
ARG RUST_VERSION=1.99.1
ARG NODE_MAJOR=26
ARG DOTNET_CHANNEL=10.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NO_COLOR=0 \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    DOTNET_ROOT=/usr/share/dotnet \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH="/usr/local/cargo/bin:/usr/local/go/bin:/usr/share/dotnet:${PATH}"

# 1. Ubuntu packages: Python 3.14 + mypy, OpenJDK 25, Git, the C linker that
#    rustc needs, and the libraries that .NET needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    mypy \
    openjdk-25-jdk-headless \
    git \
    ca-certificates \
    curl \
    xz-utils \
    gcc \
    libc6-dev \
    libicu78 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Go from go.dev, checked against the published SHA-256.
RUN set -eu; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64) sum="$GO_SHA256_AMD64" ;; \
      arm64) sum="$GO_SHA256_ARM64" ;; \
      *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-${arch}.tar.gz" -o /tmp/go.tgz; \
    echo "$sum  /tmp/go.tgz" | sha256sum -c -; \
    tar -C /usr/local -xzf /tmp/go.tgz; \
    rm /tmp/go.tgz; \
    go version

# 3. Rust through rustup (minimal profile: rustc, cargo, std).
RUN set -eu; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64) triple=x86_64-unknown-linux-gnu; sum="$RUSTUP_SHA256_AMD64" ;; \
      arm64) triple=aarch64-unknown-linux-gnu; sum="$RUSTUP_SHA256_ARM64" ;; \
      *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/${triple}/rustup-init" -o /tmp/rustup-init; \
    echo "$sum  /tmp/rustup-init" | sha256sum -c -; \
    chmod +x /tmp/rustup-init; \
    /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain "$RUST_VERSION"; \
    rm /tmp/rustup-init; \
    chmod -R a+rX "$RUSTUP_HOME" "$CARGO_HOME"; \
    rustc --version

# 4. Node.js LTS from NodeSource.
RUN curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && node --version

# 5. .NET SDK from Microsoft.
RUN curl -fsSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin --channel "$DOTNET_CHANNEL" --install-dir /usr/share/dotnet \
    && ln -sf /usr/share/dotnet/dotnet /usr/bin/dotnet \
    && dotnet --version

# 6. Pinned offline test frameworks for the graders (Tests 20-25, 32):
#    TypeScript + Vitest, JUnit + Mockito, xUnit + Moq. Copied alone so code
#    edits do not rebuild this layer.
COPY scripts/setup_polyglot_toolchains.sh /tmp/setup_polyglot_toolchains.sh
RUN bash /tmp/setup_polyglot_toolchains.sh && rm /tmp/setup_polyglot_toolchains.sh
ENV POLYGLOT_HOME=/opt/polyglot \
    NUGET_PACKAGES=/opt/polyglot/nuget \
    DOTNET_NOLOGO=1 \
    DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 \
    GOTOOLCHAIN=local \
    GOFLAGS=-mod=mod \
    GOPROXY=off

WORKDIR /app

COPY . /app

# Mount points for reports and logs.
RUN mkdir -p /app/results /app/logs
VOLUME ["/app/results", "/app/logs"]

ENTRYPOINT ["python3", "eval.py"]
CMD ["--help"]

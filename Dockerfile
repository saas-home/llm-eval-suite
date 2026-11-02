# ============================================================================
# LLM-EVAL-SUITE: Polyglot Industry-Standard LTS Evaluation Container
# Toolchains:
#   - C#: .NET 10 SDK (LTS)
#   - Java: OpenJDK 21 JDK (LTS)
#   - Node.js: Node.js 22 (Active LTS) + TypeScript (Latest Stable)
#   - Go: Go (Stable)
#   - Rust: Rust (Stable) + Cargo
#   - Python: Python 3.12 (LTS)
#
# Multi-arch compatible: x86_64 & aarch64 (ARM64 / Grace Hopper / DGX Spark)
# Zero host impact: 100% isolated inside container.
# ============================================================================

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1
ENV DOTNET_ROOT=/usr/share/dotnet
ENV PATH="/usr/share/dotnet:${PATH}"
ENV NO_COLOR=0

# 1. Base system utilities, Python 3.12 LTS, Git, OpenJDK 21 LTS, Go, Rust
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    git \
    ca-certificates \
    curl \
    libicu74 \
    openjdk-21-jdk-headless \
    golang-go \
    rustc \
    cargo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Node.js 22 LTS + Latest TypeScript
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g typescript \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Official Microsoft .NET 10 SDK (LTS)
RUN curl -sSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin --channel 10.0 --install-dir /usr/share/dotnet \
    && ln -sf /usr/share/dotnet/dotnet /usr/bin/dotnet

WORKDIR /app

# Copy benchmark suite code
COPY . /app

# Volume mount targets for benchmark results & diagnostic logs persistence
RUN mkdir -p /app/results /app/logs
VOLUME ["/app/results", "/app/logs"]

ENTRYPOINT ["python3", "eval.py"]
CMD ["--help"]

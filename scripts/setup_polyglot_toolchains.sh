#!/usr/bin/env bash
# Installs pinned, offline test dependencies for the polyglot execution
# graders (Tests 20-25) into /opt/polyglot. Run once at image build time;
# grading then works without network access.
#
#   TypeScript : vitest + typescript + @types/node (local node_modules)
#   Java       : JUnit console launcher + Mockito (jars); javatuples for MultiPL-E
#   C#         : xUnit v3 + Moq restored into a local NuGet package folder,
#                plus pre-restored console/test project templates
#   Go, Rust   : standard toolchain only (testing.T / #[test])
set -euo pipefail

ROOT=/opt/polyglot
mkdir -p "$ROOT"

# --- TypeScript ---------------------------------------------------------
mkdir -p "$ROOT/ts"
cat > "$ROOT/ts/package.json" <<'EOF'
{
  "name": "polyglot-ts",
  "private": true,
  "devDependencies": {
    "@types/node": "26.6.6",
    "typescript": "7.0.2",
    "vitest": "5.1.0"
  }
}
EOF
(cd "$ROOT/ts" && npm install --no-audit --no-fund --loglevel=error)

# --- Java ---------------------------------------------------------------
mkdir -p "$ROOT/java/lib"
MAVEN=https://repo1.maven.org/maven2
# Pinned SHA-256 per jar (checked against Maven Central's published SHA-1 on
# 2026-11-07). A download that does not match stops the image build.
while read -r sha artifact; do
  jar="$ROOT/java/lib/$(basename "$artifact")"
  curl -fsSL "$MAVEN/$artifact" -o "$jar"
  echo "$sha  $jar" | sha256sum -c --quiet -
done <<'EOF'
e62b96ac475dbcde8599ea905d088f65d90778f86e259b856a49fa5c4ea256ec org/junit/platform/junit-platform-console-standalone/6.1.3/junit-platform-console-standalone-6.1.3.jar
97a1f835ad84075255aa37809e4379eff541273c02c6201cbad49bd5950932cd org/mockito/mockito-core/5.24.0/mockito-core-5.24.0.jar
904d4ba9769284739dd720eceffe3792a67e9c0e78d8a50ad46925283386e048 org/mockito/mockito-junit-jupiter/5.24.0/mockito-junit-jupiter-5.24.0.jar
bb78cc9d290d13e27e86fe89fb171521516a33c9822a528c0f151bece5790276 net/bytebuddy/byte-buddy/1.18.17/byte-buddy-1.18.17.jar
4d72f491b272f461d9476018aada019fa366ddb07a6feece9340bb5b5ab64bdd net/bytebuddy/byte-buddy-agent/1.18.17/byte-buddy-agent-1.18.17.jar
6ecc0fc776af04e2bfd2bf42d618ab5161440dcb8a0e7dff9e84c065c7cec3dd org/objenesis/objenesis/3.6/objenesis-3.6.jar
2eda5b19d9820e1cc2f69fcd01639a715a673c11f8507e3d1ed593cf765d5e0a org/javatuples/javatuples/1.2/javatuples-1.2.jar
EOF

# --- C# -----------------------------------------------------------------
export DOTNET_CLI_HOME=/tmp/dotnet-home
export DOTNET_NOLOGO=1
export DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
export NUGET_PACKAGES="$ROOT/nuget"
mkdir -p "$ROOT/csharp/app" "$ROOT/csharp/tests" "$NUGET_PACKAGES"

cat > "$ROOT/csharp/app/App.csproj" <<'EOF'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>
</Project>
EOF
cat > "$ROOT/csharp/tests/Tests.csproj" <<'EOF'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <IsPackable>false</IsPackable>
    <OutputType>Exe</OutputType>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="18.11.0" />
    <PackageReference Include="xunit.v3" Version="4.0.2" />
    <PackageReference Include="xunit.runner.visualstudio" Version="4.0.1" />
    <PackageReference Include="Moq" Version="4.21.0" />
  </ItemGroup>
</Project>
EOF
echo 'Console.WriteLine("ok");' > "$ROOT/csharp/app/Program.cs"
echo 'public class Smoke { [Xunit.Fact] public void Ok() { } }' > "$ROOT/csharp/tests/Smoke.cs"
(cd "$ROOT/csharp/app" && dotnet build -v q --nologo)
# xUnit v3 test projects are executables; polyglot.py runs them the same way.
(cd "$ROOT/csharp/tests" && dotnet build -v q --nologo -o bin_out && dotnet bin_out/Tests.dll)
rm -rf "$ROOT/csharp/app/bin" "$ROOT/csharp/app/obj" "$ROOT/csharp/tests/bin" "$ROOT/csharp/tests/obj" "$ROOT/csharp/tests/bin_out"
rm -f "$ROOT/csharp/app/Program.cs" "$ROOT/csharp/tests/Smoke.cs"

chmod -R a+rX "$ROOT"
echo "polyglot toolchains ready in $ROOT"

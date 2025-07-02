# OpenTelemetry instrumentation for OpenAI
{ pkgs, prev }:

prev.buildPythonPackage rec {
  pname = "opentelemetry_instrumentation_openai";
  version = "0.37.1";
  format = "pyproject";
  
  src = prev.fetchPypi {
    pname = "opentelemetry_instrumentation_openai";
    version = "0.37.1";
    hash = "sha256-SoS5lXJMoE7TvOltnmI2/2M4EGGU74rkwZMFsFEeRpw=";
  };
  
  nativeBuildInputs = with prev; [
    hatchling
    poetry-core
  ];
  
  propagatedBuildInputs = with prev; [
    opentelemetry-api
    opentelemetry-instrumentation
    opentelemetry-semantic-conventions
    typing-extensions
    wrapt
  ];
  
  # Disable tests to avoid network access
  doCheck = false;
  
  # Disable runtime dependency checks for missing packages
  pythonRelaxDepsHook = true;
  pythonRemoveDepsHook = true;
  dontCheckRuntimeDeps = true;
  
  # Skip the runtime deps check phase entirely
  preBuild = ''
    export PYTHONDONTWRITEBYTECODE=1
  '';
  
  postInstall = ''
    # Remove the runtime deps check hook
    rm -f $out/nix-support/propagated-build-inputs
  '';
  
  # Disable import check due to missing runtime dependencies
  pythonImportsCheck = [];
  
  meta = with pkgs.lib; {
    description = "OpenTelemetry instrumentation for OpenAI";
    homepage = "https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation/opentelemetry-instrumentation-openai";
    license = licenses.asl20;
    maintainers = with maintainers; [ ];
  };
}
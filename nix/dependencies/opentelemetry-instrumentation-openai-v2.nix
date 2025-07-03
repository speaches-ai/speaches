# OpenTelemetry instrumentation for OpenAI v2
{
  pkgs,
  prev,
}:
prev.buildPythonPackage rec {
  pname = "opentelemetry_instrumentation_openai_v2";
  version = "2.1b0";
  format = "pyproject";

  src = prev.fetchPypi {
    pname = "opentelemetry_instrumentation_openai_v2";
    version = "2.1b0";
    hash = "sha256-GEqV+Ewo9Xn7zXixtULTz3XmvR3Jw7jHvkeGoZzbrxM=";
  };

  nativeBuildInputs = with prev; [
    hatchling
    poetry-core
  ];

  propagatedBuildInputs = with prev; [
    opentelemetry-api
    opentelemetry-instrumentation
    opentelemetry-semantic-conventions
    httpx
    wrapt
  ];

  # Disable tests to avoid network access
  doCheck = false;

  # Disable import check due to complex dependencies
  pythonImportsCheck = [];

  meta = with pkgs.lib; {
    description = "OpenTelemetry instrumentation for OpenAI v2";
    homepage = "https://github.com/open-telemetry/opentelemetry-python-contrib";
    license = licenses.asl20;
    maintainers = with maintainers; [];
  };
}


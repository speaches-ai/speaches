# OpenTelemetry instrumentation packages for speaches
{pyPackages}: {
  opentelemetry_instrumentation_openai = pyPackages.buildPythonPackage {
    pname = "opentelemetry_instrumentation_openai";
    version = "0.37.1";
    format = "pyproject";
    src = pyPackages.fetchPypi {
      pname = "opentelemetry_instrumentation_openai";
      version = "0.37.1";
      hash = "sha256-SoS5lXJMoE7TvOltnmI2/2M4EGGU74rkwZMFsFEeRpw=";
    };
    nativeBuildInputs = with pyPackages; [hatchling poetry-core];
    propagatedBuildInputs = with pyPackages; [
      opentelemetry-api
      opentelemetry-instrumentation
      opentelemetry-semantic-conventions
      typing-extensions
      wrapt
    ];
    doCheck = false;
    dontCheckRuntimeDeps = true;
    pythonImportsCheck = [];
  };

  opentelemetry_instrumentation_openai_v2 = pyPackages.buildPythonPackage {
    pname = "opentelemetry_instrumentation_openai_v2";
    version = "2.1b0";
    format = "pyproject";
    src = pyPackages.fetchPypi {
      pname = "opentelemetry_instrumentation_openai_v2";
      version = "2.1b0";
      hash = "sha256-GEqV+Ewo9Xn7zXixtULTz3XmvR3Jw7jHvkeGoZzbrxM=";
    };
    nativeBuildInputs = with pyPackages; [hatchling poetry-core];
    propagatedBuildInputs = with pyPackages; [
      opentelemetry-api
      opentelemetry-instrumentation
      opentelemetry-semantic-conventions
      httpx
      wrapt
    ];
    doCheck = false;
    dontCheckRuntimeDeps = true;
    pythonImportsCheck = [];
  };
}

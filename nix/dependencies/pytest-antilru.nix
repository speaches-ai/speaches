# pytest-antilru - Pytest plugin for antilru
{
  pkgs,
  prev,
}:
prev.buildPythonPackage rec {
  pname = "pytest_antilru";
  version = "2.0.0";
  format = "pyproject";

  src = prev.fetchPypi {
    inherit pname version;
    hash = "sha256-SM/zQmSLahzk5TmM8gOWaQXVRrPyvue7VdfLPsh6hfs=";
  };

  nativeBuildInputs = with prev; [
    poetry-core
  ];

  propagatedBuildInputs = with prev; [
    pytest
  ];

  # Disable tests to avoid circular dependency
  doCheck = false;

  pythonImportsCheck = ["pytest_antilru"];

  meta = with pkgs.lib; {
    description = "Pytest plugin for antilru";
    homepage = "https://github.com/pytest-dev/pytest-antilru";
    license = licenses.mit;
    maintainers = with maintainers; [];
  };
}


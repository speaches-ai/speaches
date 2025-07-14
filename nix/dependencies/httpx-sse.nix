# httpx-sse - Server-sent events support for HTTPX
{ pkgs, prev }:

prev.buildPythonPackage rec {
  pname = "httpx_sse";
  version = "0.4.0";
  format = "setuptools";
  
  src = prev.fetchPypi {
    pname = "httpx-sse";
    inherit version;
    hash = "sha256-HoGjowcM4yKt0dNSntQutfcIF/Re1uyRWrdT+WETlyE=";
  };
  
  propagatedBuildInputs = with prev; [
    httpx
    httpcore
  ];
  
  # Disable tests to avoid network access
  doCheck = false;
  
  pythonImportsCheck = [ "httpx_sse" ];
  
  meta = with pkgs.lib; {
    description = "Server-sent events support for HTTPX";
    homepage = "https://github.com/florimondmanca/httpx-sse";
    license = licenses.bsd3;
    maintainers = with maintainers; [ ];
  };
}
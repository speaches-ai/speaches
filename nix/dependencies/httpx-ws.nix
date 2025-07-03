# httpx-ws - WebSockets support for HTTPX
{ pkgs, prev }:

prev.buildPythonPackage rec {
  pname = "httpx_ws";
  version = "0.7.2";
  format = "setuptools";
  
  src = prev.fetchPypi {
    pname = "httpx_ws";  # Use underscores to match the actual filename
    inherit version;
    hash = "sha256-k+3qbI/DE0ZPwoe/99KtIOYZa3dUx2+Ub3O0r3mIbU4=";
  };
  
  propagatedBuildInputs = with prev; [
    httpx
    httpcore
    wsproto
    anyio
  ];
  
  # Disable tests to avoid network access
  doCheck = false;
  
  pythonImportsCheck = [ "httpx_ws" ];
  
  meta = with pkgs.lib; {
    description = "WebSockets support for HTTPX";
    homepage = "https://github.com/frankie567/httpx-ws";
    license = licenses.mit;
    maintainers = with maintainers; [ ];
  };
}
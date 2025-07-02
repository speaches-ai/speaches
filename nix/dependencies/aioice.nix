# aioice - asyncio-based Interactive Connectivity Establishment (RFC 5245)
{ pkgs, prev }:

prev.buildPythonPackage rec {
  pname = "aioice";
  version = "0.9.0";
  format = "setuptools";
  
  src = prev.fetchPypi {
    inherit pname version;
    hash = "sha256-/CQBscS24ZNy6q6qKP0b2cv2sOQS5IYlKXxTtJXuvR4=";
  };
  
  propagatedBuildInputs = with prev; [
    dnspython
    ifaddr
  ];
  
  # Disable tests as they require network access
  doCheck = false;
  
  pythonImportsCheck = [ "aioice" ];
  
  meta = with pkgs.lib; {
    description = "asyncio-based Interactive Connectivity Establishment (RFC 5245)";
    homepage = "https://github.com/aiortc/aioice";
    license = licenses.bsd3;
  };
}
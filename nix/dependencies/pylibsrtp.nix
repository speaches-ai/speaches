# pylibsrtp - Python bindings for libsrtp
{ pkgs, prev }:

prev.buildPythonPackage rec {
  pname = "pylibsrtp";
  version = "0.10.0";
  format = "setuptools";
  
  src = prev.fetchPypi {
    inherit pname version;
    hash = "sha256-2AAZEtf1G9BbTqNVF0eTBjF3f9N4ks87/g5UGnQuaZ8=";
  };
  
  nativeBuildInputs = with prev; [
    cffi
  ];
  
  buildInputs = with pkgs; [
    srtp
    openssl
  ];
  
  propagatedBuildInputs = with prev; [
    cffi
  ];
  
  # Disable tests
  doCheck = false;
  
  pythonImportsCheck = [ "pylibsrtp" ];
  
  meta = with pkgs.lib; {
    description = "Python bindings for libsrtp";
    homepage = "https://github.com/aiortc/pylibsrtp";
    license = licenses.bsd3;
  };
}
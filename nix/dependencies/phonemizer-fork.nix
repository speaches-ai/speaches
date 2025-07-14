# phonemizer-fork - Alternative phonemizer implementation
{ pkgs, prev }:

prev.buildPythonPackage rec {
  pname = "phonemizer-fork";
  version = "3.3.2";
  format = "wheel";
  
  src = pkgs.fetchurl {
    url = "https://files.pythonhosted.org/packages/64/f1/0dcce21b0ae16a82df4b6583f8f3ad8e55b35f7e98b6bf536a4dd225fa08/phonemizer_fork-3.3.2-py3-none-any.whl";
    hash = "sha256-lzBcdvQYOzgl2uj0wDImX+eMmUbOWMR9S2IWE0kmS3Q=";
  };
  
  propagatedBuildInputs = with prev; [
    joblib
    segments
    attrs
    dlinfo
  ];
  
  doCheck = false;
  
  meta = with pkgs.lib; {
    description = "Simple text to phonemes converter for multiple languages";
    homepage = "https://github.com/bootphon/phonemizer";
    license = licenses.gpl3;
  };
}
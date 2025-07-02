# piper-tts package (Linux only)
{ pkgs, prev, system }:

if pkgs.stdenv.isLinux then
  prev.buildPythonPackage rec {
    pname = "piper_tts";
    version = "1.2.0";
    format = "wheel";
    
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/24/aa/215bced0725cf5b5afe939f86b177c8ddb0d38292a94e85c55b3fcf6d46d/piper_tts-1.2.0-py3-none-any.whl";
      hash = "sha256-80EK6g+AUdihGAUKW5VPrrNvasDabRD8L4oEOo6vJ7U=";
    };
    
    propagatedBuildInputs = with prev; [
      (import ./piper-phonemize.nix { inherit pkgs prev system; })
    ];
    
    doCheck = false;
    
    meta = with pkgs.lib; {
      description = "A fast, local neural text to speech system";
      homepage = "https://github.com/rhasspy/piper";
      license = licenses.mit;
      platforms = platforms.linux;
    };
  }
else null
# piper-phonemize package (Linux only, architecture-specific wheels)
{ pkgs, prev, system }:

if system == "x86_64-linux" then
  prev.buildPythonPackage rec {
    pname = "piper_phonemize";
    version = "1.2.0";
    format = "wheel";
    
    src = pkgs.fetchurl {
      url = "https://github.com/fedirz/piper-phonemize/raw/refs/heads/master/dist/piper_phonemize-1.2.0-cp312-cp312-manylinux_2_28_x86_64.whl";
      hash = "sha256-E7/QdVBXIELF5t2NQAdr8kEBqTCvDHoSZUJyFydSJbM=";
    };
    
    doCheck = false;
    
    meta = with pkgs.lib; {
      description = "Phonemization library for Piper TTS";
      homepage = "https://github.com/rhasspy/piper-phonemize";
      license = licenses.mit;
      platforms = [ "x86_64-linux" ];
    };
  }
else if system == "aarch64-linux" then
  prev.buildPythonPackage rec {
    pname = "piper_phonemize";
    version = "1.2.0";
    format = "wheel";
    
    src = pkgs.fetchurl {
      url = "https://github.com/fedirz/piper-phonemize/raw/refs/heads/master/dist/piper_phonemize-1.2.0-cp312-cp312-manylinux_2_28_aarch64.whl";
      hash = "sha256-0000000000000000000000000000000000000000000="; # TODO: Calculate actual hash
    };
    
    doCheck = false;
    
    meta = with pkgs.lib; {
      description = "Phonemization library for Piper TTS";
      homepage = "https://github.com/rhasspy/piper-phonemize";
      license = licenses.mit;
      platforms = [ "aarch64-linux" ];
    };
  }
else null
# Piper TTS packages for speaches (Linux-only)
{
  pkgs,
  pyPackages,
  system,
}: let
  # Only build on x86_64-linux
  isLinuxX86 = system == "x86_64-linux";
in rec {
  piper_phonemize =
    if isLinuxX86
    then
      pyPackages.buildPythonPackage {
        pname = "piper_phonemize";
        version = "1.2.0";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://github.com/fedirz/piper-phonemize/raw/refs/heads/master/dist/piper_phonemize-1.2.0-cp312-cp312-manylinux_2_28_x86_64.whl";
          hash = "sha256-E7/QdVBXIELF5t2NQAdr8kEBqTCvDHoSZUJyFydSJbM=";
        };
        doCheck = false;
      }
    else null;

  piper_tts =
    if pkgs.stdenv.isLinux && isLinuxX86
    then
      pyPackages.buildPythonPackage rec {
        pname = "piper_tts";
        version = "1.2.0";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/24/aa/215bced0725cf5b5afe939f86b177c8ddb0d38292a94e85c55b3fcf6d46d/piper_tts-1.2.0-py3-none-any.whl";
          hash = "sha256-80EK6g+AUdihGAUKW5VPrrNvasDabRD8L4oEOo6vJ7U=";
        };
        propagatedBuildInputs = [
          # Reference the piper_phonemize from this same overlay using rec
          piper_phonemize
        ];
        doCheck = false;
      }
    else null;
}

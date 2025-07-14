# espeakng-loader - Python loader for espeak-ng
{ pkgs, prev }:

prev.buildPythonPackage {
  pname = "espeakng_loader";
  version = "0.1.0";
  format = "pyproject";
  
  src = pkgs.fetchFromGitHub {
    owner = "thewh1teagle";
    repo = "espeakng-loader";
    rev = "main";
    hash = "sha256-YlqlC5/x54y2nz2o4InCXOy802VE2VEDl7SRr3sBcTk=";
  };
  
  nativeBuildInputs = with prev; [
    setuptools
    wheel
    pip
    hatchling
  ];
  
  propagatedBuildInputs = [ pkgs.espeak-ng ];
  
  # Patch the library to use Nix paths
  postPatch = ''
    substituteInPlace src/espeakng_loader/__init__.py \
      --replace 'libespeak-ng.so' '${pkgs.espeak-ng}/lib/libespeak-ng.so' \
      --replace 'libespeak-ng.so.1' '${pkgs.espeak-ng}/lib/libespeak-ng.so.1' \
      --replace 'libespeak-ng' '${pkgs.espeak-ng}/lib/libespeak-ng' \
      --replace "data_path = Path(__file__).parent / 'espeak-ng-data'" "data_path = Path('${pkgs.espeak-ng}/share/espeak-ng-data')"
  '';
  
  doCheck = false;
  
  meta = with pkgs.lib; {
    description = "Python loader for espeak-ng";
    homepage = "https://github.com/thewh1teagle/espeakng-loader";
    license = licenses.mit;
  };
}
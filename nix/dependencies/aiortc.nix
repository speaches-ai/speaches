# aiortc - WebRTC and ORTC implementation for Python using asyncio
{ pkgs, prev, aioice, pylibsrtp }:

prev.buildPythonPackage rec {
  pname = "aiortc";
  version = "1.9.0";
  format = "setuptools";
  
  src = prev.fetchPypi {
    inherit pname version;
    hash = "sha256-A/qnbXbvDlmJrBA4aJiwKTaXVhAiFyMOL81LApxQswM=";
  };
  
  nativeBuildInputs = with prev; [
    setuptools
    wheel
  ];
  
  propagatedBuildInputs = with prev; [
    pyee
    pyopenssl
    cryptography
    av
    dnspython
    ifaddr
    google-crc32c
  ] ++ [
    # Custom packages passed as parameters
    aioice
    pylibsrtp
  ];
  
  buildInputs = with pkgs; [
    ffmpeg-full
    libvpx
    libopus
    srtp
  ];
  
  # Disable tests as they require network access
  doCheck = false;
  
  pythonImportsCheck = [ "aiortc" ];
  
  meta = with pkgs.lib; {
    description = "WebRTC and ORTC implementation for Python using asyncio";
    homepage = "https://github.com/aiortc/aiortc";
    license = licenses.bsd3;
    maintainers = with maintainers; [ ];
  };
}
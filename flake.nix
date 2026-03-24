{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-master.url = "github:NixOS/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        linuxLibPath =
          if system == "x86_64-linux" then
            pkgs.lib.makeLibraryPath [
              # Needed for `soundfile`
              pkgs.portaudio

              pkgs.zlib
              pkgs.stdenv.cc.cc
              pkgs.openssl
            ]
          else
            "";

      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            docker
            docker-compose
            ffmpeg-full
            go-task
            python312
            uv
          ];

          LD_LIBRARY_PATH = linuxLibPath;

          shellHook = ''
            source .venv/bin/activate
            source .env
          '';
        };

        formatter = pkgs.nixfmt;
      }
    );
}

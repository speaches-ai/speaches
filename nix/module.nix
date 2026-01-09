{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.speaches;
in
{
  options.services.speaches = {
    enable = lib.mkEnableOption "speaches AI speech processing service";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.speaches;
      defaultText = lib.literalExpression "pkgs.speaches";
      description = "The speaches package to use";
    };

    pythonVersion = lib.mkOption {
      type = lib.types.enum [
        "python311"
        "python312"
        "python313"
      ];
      default = "python312";
      description = "Python version to use";
    };

    enablePiper = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable Piper TTS support";
    };

    enableOtel = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable OpenTelemetry instrumentation";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8000;
      description = "Port to listen on";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "0.0.0.0";
      description = "Host to bind to";
    };

    enableCuda = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable CUDA support";
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Environment file containing secrets";
    };

    huggingFaceHome = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Environment file containing secrets";
    };

    dataDir = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/speaches";
      description = "Directory for speaches data and cache";
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "Extra command line arguments to pass to speaches";
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "Additional environment variables for the service";
    };
  };

  config = lib.mkIf cfg.enable (
    let
      hfHome = if cfg.huggingFaceHome != null then cfg.huggingFaceHome else "${cfg.dataDir}/huggingface";
    in
    {
      systemd.services.speaches = {
        description = "Speaches AI speech processing service";
        after = [ "network.target" ];
        wantedBy = [ "multi-user.target" ];

        environment = {
          UVICORN_HOST = cfg.host;
          UVICORN_PORT = toString cfg.port;
          HF_HUB_ENABLE_HF_TRANSFER = "0";
          DO_NOT_TRACK = "1";
          HF_HOME = hfHome;
        }
        // cfg.environment;

        serviceConfig = {
          Type = "exec";
          ExecStart = "${cfg.package}/bin/speaches ${lib.escapeShellArgs cfg.extraArgs}";
          Restart = "on-failure";
          RestartSec = 5;
          User = "speaches";
          Group = "speaches";
          WorkingDirectory = cfg.dataDir;

          # Security hardening
          NoNewPrivileges = true;
          PrivateTmp = true;
          ProtectSystem = "strict";
          ProtectHome = true;
          ReadWritePaths = [ cfg.dataDir ];

          # Load environment file if specified
          EnvironmentFile = lib.mkIf (cfg.environmentFile != null) cfg.environmentFile;
        };

        preStart = ''
          mkdir -p ${hfHome}/hub
        '';
      };

      # Create user and group
      users.users.speaches = {
        isSystemUser = true;
        group = "speaches";
        home = cfg.dataDir;
        createHome = true;
      };

      users.groups.speaches = { };

      # Open firewall port
      networking.firewall.allowedTCPPorts = [ cfg.port ];
    }
  );
}

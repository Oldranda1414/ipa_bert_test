{
  description = "Dev environment for uv project";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
  };

  outputs = { self , nixpkgs ,... }: let
    system = "x86_64-linux";
  in {
    devShells."${system}".default = let
      pkgs = import nixpkgs {
        inherit system;
      };
    in pkgs.mkShell {
      packages = with pkgs; [
        # modern python package manager
        uv
        # phonemizer backend
        espeak-classic
      ];

      shellHook = ''
        export PATH="${pkgs.espeak-classic}/bin:$PATH"
        # Add espeak libraries to the library path
        export LD_LIBRARY_PATH="${pkgs.espeak-classic}/lib:$LD_LIBRARY_PATH"
      '';
    };
  };
}

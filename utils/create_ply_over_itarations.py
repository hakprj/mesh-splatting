#!/usr/bin/env python3
# triangles_to_ply.py
import argparse, os, sys
import numpy as np
import torch
import trimesh

def parse_vec3(s):
    try:
        x, y, z = [float(v) for v in s.split(",")]
        return torch.tensor([x, y, z], dtype=torch.float32)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid vec3 '{s}', expected 'x,y,z'") from e

def load_scene(scene_dir, device="cpu"):
    path = os.path.join(scene_dir, "point_cloud_state_dict.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find '{path}'")
    sd = torch.load(path, map_location=device)

    # base mesh
    verts   = sd["triangles_points"].to(device).to(torch.float32)   # [V,3]
    faces   = sd["_triangle_indices"].to(device).to(torch.int64)    # [T,3]
    f_dc    = sd["features_dc"].to(device).to(torch.float32)        # [V,1,3]
    f_rest  = sd.get("features_rest", None)
    if f_rest is not None:
        f_rest = f_rest.to(device).to(torch.float32)                # [V,?,3] (unused here)
    act_deg = int(sd.get("active_sh_degree", 3))

    return verts, faces, f_dc, f_rest, act_deg


def export_ply(out_path, verts_np, faces_np, colors_rgb, zup_to_yup=False):
    if zup_to_yup:
        # Create a copy to avoid modifying the original data
        transformed_verts = verts_np.copy()
        
        # Apply transformation: (x, y, z) -> (x, z, -y)
        # This should convert from Z-up to Y-up
        transformed_verts[:, 1] = verts_np[:, 2]   # y = z
        transformed_verts[:, 2] = -verts_np[:, 1]  # z = -y
        
        verts_np = transformed_verts

    # Create mesh with vertex colors
    mesh = trimesh.Trimesh(vertices=verts_np.astype(np.float32),
                           faces=faces_np.astype(np.int32),
                           vertex_colors=colors_rgb.astype(np.uint8),
                           process=False)
    
    # Export as PLY
    mesh.export(out_path, file_type='ply')

def main():
    p = argparse.ArgumentParser(description="Export triangle scene to PLY with per-vertex colors.")
    p.add_argument("scene_dir", nargs='?', type=str, default=None, help="Directory containing point_cloud_state_dict.pt")
    p.add_argument("--out", type=str, default="mesh.ply", help="Output path, default <scene_dir>/mesh.ply")
    p.add_argument("--camera", type=parse_vec3, default="0,0,0", help="Camera center x,y,z")
    p.add_argument("--degree", type=int, default=None, help="Override SH degree to evaluate")
    p.add_argument("--cpu", action="store_true", help="Force CPU for loading and SH eval")
    p.add_argument("--zup", action="store_true", help="Input is Z-up, rotate to Y-up for standard viewers")
    p.add_argument("--checkpoint-dir", type=str, help="Directory with iteration checkpoints (e.g., output/xyz)")
    args = p.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint_dir:
        # Iterate over checkpoints
        checkpoint_dir = args.checkpoint_dir
        if not os.path.isdir(checkpoint_dir):
            print(f"Error: checkpoint directory '{checkpoint_dir}' not found", file=sys.stderr)
            sys.exit(1)

        # Find all iteration directories matching pattern: iteration_XXX
        iterations = sorted([d for d in os.listdir(checkpoint_dir) 
                           if d.startswith("iteration_") and os.path.isdir(os.path.join(checkpoint_dir, d))])

        if not iterations:
            print(f"Warning: No iteration_* directories found in {checkpoint_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(iterations)} iterations: {iterations}")
        
        for iteration_dir in iterations:
            scene_path = os.path.join(checkpoint_dir, iteration_dir)
            try:
                verts, faces, f_dc, f_rest, act_deg = load_scene(scene_path, device=device)
                camera_center = args.camera.to(device)

                SH_C0 = 0.28209479177387814
                colors = SH_C0 * f_dc + 0.5
                colors = torch.clamp(colors, 0.0, 1.0)
                colors_u8 = (colors * 255.0).round().to(torch.uint8).cpu().numpy()
                colors_u8 = colors_u8.squeeze()

                verts_np = verts.detach().cpu().numpy()
                faces_np = faces.detach().cpu().numpy()

                out_path = os.path.join(checkpoint_dir, f"{iteration_dir}_mesh.ply")
                export_ply(out_path, verts_np, faces_np, colors_u8, zup_to_yup=args.zup)
                print(f"Saved PLY to: {out_path} (V: {verts_np.shape[0]}, F: {faces_np.shape[0]})")
            except Exception as e:
                print(f"Warning: Failed to process {iteration_dir}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                continue
    else:
        # Single scene export (original behavior)
        if not args.scene_dir:
            print("Error: must provide scene_dir or --checkpoint-dir", file=sys.stderr)
            sys.exit(1)
            
        verts, faces, f_dc, f_rest, act_deg = load_scene(args.scene_dir, device=device)
        camera_center = args.camera.to(device)

        SH_C0 = 0.28209479177387814
        colors = SH_C0 * f_dc + 0.5
        colors = torch.clamp(colors, 0.0, 1.0)
        colors_u8 = (colors * 255.0).round().to(torch.uint8).cpu().numpy()
        colors_u8 = colors_u8.squeeze()

        verts_np = verts.detach().cpu().numpy()
        faces_np = faces.detach().cpu().numpy()

        out_path = args.out or os.path.join(args.scene_dir, "mesh.ply")
        
        export_ply(out_path, verts_np, faces_np, colors_u8, zup_to_yup=args.zup)
        print(f"Saved PLY to: {out_path}")
        print(f"Vertices: {verts_np.shape[0]}, Faces: {faces_np.shape[0]}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
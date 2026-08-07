"""
Generate Comprehensive 3D Inference Visualizations for Optimization Experiments

Creates multiplane, rotation_3d, and slices visualizations for each optimization experiment.
Similar structure to 3D_Project_Inference_Image/Multi_Organ_Segmentation/

Output structure:
Optimization_3D_Project_Inference/
├── box_point/
│   ├── box10_point90/
│   │   ├── multiplane/
│   │   ├── rotation_3d/
│   │   └── slices/
│   └── ...
├── boxpoint_box_point/
│   └── ...
└── ucb1_global/
    └── ...
"""

import sys
from pathlib import Path

from bapmos.paths import find_combined_masks_dir, project_root, simulation_dataset_dir

_3d_ext = project_root() / "3D_Project_Inference"
if _3d_ext.is_dir():
    sys.path.insert(0, str(_3d_ext))

import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

from comparison_calculator import ComparisonCalculator
from volume_overlay_renderer import VolumeOverlayRenderer
from slice_comparison_renderer import SliceComparisonRenderer


ORGAN_NAMES = {
    1: 'Rectum',
    2: 'Bladder',
    3: 'PTV1'
}

# Map combination names to class IDs that should be included
COMBINATION_CLASSES = {
    'Rectum': [1],
    'Bladder': [2],
    'PTV1+Rectum': [1, 3],
    'Bladder+Rectum': [1, 2],
    'none': []
}


class Optimization3DInferenceGenerator:
    """Generate comprehensive 3D visualizations for optimization experiments."""
    
    def __init__(self, viz_base: Path, output_base: Path):
        self.viz_base = viz_base  # Optimization_Visualization/
        self.output_base = output_base  # Optimization_3D_Project_Inference/
        self.gt_base = find_combined_masks_dir(simulation_dataset_dir())
        
        self.calculator = ComparisonCalculator()
        self.volume_renderer = VolumeOverlayRenderer()
        self.slice_renderer = SliceComparisonRenderer()
        
        # Load organ combinations
        combo_file = self.gt_base / "organ_combinations.json"
        with open(combo_file, 'r') as f:
            combo_data = json.load(f)
            self.organ_combinations = combo_data['combinations']
        
        print(f"Loaded organ combinations:")
        for combo, imgs in self.organ_combinations.items():
            print(f"  {combo}: {len(imgs)} images")
    
    def load_predictions_from_all_splits(self, exp_path: Path):
        """Load predictions from all splits (train, val, test)."""
        predictions = {}
        
        for split in ['train', 'val', 'test']:
            pred_dir = exp_path / split / "predictions"
            
            if not pred_dir.exists():
                continue
            
            for pred_file in sorted(pred_dir.glob("*.png")):
                # Remove .png extension(s) - handle .png.png case
                img_id = pred_file.stem
                if img_id.endswith('.png'):
                    img_id = img_id[:-4]  # Remove .png from stem
                
                # Load and resize
                mask = np.array(Image.open(pred_file))
                mask_resized = np.array(Image.fromarray(mask).resize((256, 256), Image.NEAREST))
                # Binarize (predictions are saved as 0/255)
                predictions[img_id] = (mask_resized > 0).astype(np.uint8)
        
        return predictions
    
    def load_predictions_from_split(self, exp_path: Path, split: str):
        """Load predictions from a specific split."""
        pred_dir = exp_path / split / "predictions"
        
        if not pred_dir.exists():
            return {}
        
        predictions = {}
        for pred_file in sorted(pred_dir.glob("*.png")):
            # Remove .png extension(s) - handle .png.png case
            img_id = pred_file.stem
            if img_id.endswith('.png'):
                img_id = img_id[:-4]  # Remove .png from stem
            
            # Load and resize
            mask = np.array(Image.open(pred_file))
            mask_resized = np.array(Image.fromarray(mask).resize((256, 256), Image.NEAREST))
            # Binarize (predictions are saved as 0/255)
            predictions[img_id] = (mask_resized > 0).astype(np.uint8)
        
        return predictions
    
    def load_ground_truths(self, image_ids):
        """Load ground truth masks."""
        ground_truths = {}
        
        for img_id in image_ids:
            gt_file = self.gt_base / f"{img_id}_combined_mask.png"
            if gt_file.exists():
                gt_mask = np.array(Image.open(gt_file))
                gt_resized = np.array(Image.fromarray(gt_mask).resize((256, 256), Image.NEAREST))
                # Convert to binary (any organ = 1)
                ground_truths[img_id] = (gt_resized > 0).astype(np.uint8)
        
        return ground_truths
    
    def load_multiclass_ground_truths(self, image_ids):
        """Load ground truth masks as multi-class."""
        ground_truths = {}
        
        for img_id in image_ids:
            gt_file = self.gt_base / f"{img_id}_combined_mask.png"
            if gt_file.exists():
                gt_mask = np.array(Image.open(gt_file))
                gt_resized = np.array(Image.fromarray(gt_mask).resize((256, 256), Image.NEAREST))
                ground_truths[img_id] = gt_resized.astype(np.uint8)
        
        return ground_truths
    
    def generate_per_organ_multiplane(self, pred_volume, gt_volume, output_dir, exp_name):
        """Generate multiplane views for each organ separately."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get unique organs from ground truth
        unique_organs = np.unique(gt_volume)
        unique_organs = unique_organs[unique_organs > 0]  # Exclude background
        
        for organ_id in unique_organs:
            organ_name = ORGAN_NAMES.get(organ_id, f"Organ_{organ_id}")
            
            # Extract organ-specific masks
            organ_pred = (pred_volume == organ_id).astype(np.uint8)
            organ_gt = (gt_volume == organ_id).astype(np.uint8)
            
            # Calculate comparison
            correct, false, missed = self.calculator.calculate_volume_comparison(organ_pred, organ_gt)
            
            # Generate multiplane
            output_path = output_dir / f"{exp_name}_{organ_name}_multiplane.png"
            self.slice_renderer.render_multiplane_comparison(
                organ_gt, correct, false, missed,
                output_path=output_path,
                title=f"{exp_name} - {organ_name}"
            )
        
        # Also generate combined view
        combined_correct, combined_false, combined_missed = self.calculator.calculate_volume_comparison(
            (pred_volume > 0).astype(np.uint8),
            (gt_volume > 0).astype(np.uint8)
        )
        
        output_path = output_dir / f"{exp_name}_All_Organs_multiplane.png"
        self.slice_renderer.render_multiplane_comparison(
            (gt_volume > 0).astype(np.uint8),
            combined_correct,
            combined_false,
            combined_missed,
            output_path=output_path,
            title=f"{exp_name} - All Organs"
        )
    
    def generate_per_organ_multiplane_binary(self, pred_binary, gt_multiclass, output_dir, exp_name):
        """Generate multiplane comparing binary prediction against each GT organ."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        unique_organs = np.unique(gt_multiclass)
        unique_organs = unique_organs[unique_organs > 0]
        
        for organ_id in unique_organs:
            organ_name = ORGAN_NAMES.get(organ_id, f"Organ_{organ_id}")
            organ_gt = (gt_multiclass == organ_id).astype(np.uint8)
            
            # Compare binary prediction against this organ's GT
            correct, false, missed = self.calculator.calculate_volume_comparison(pred_binary, organ_gt)
            
            output_path = output_dir / f"{exp_name}_{organ_name}_multiplane.png"
            self.slice_renderer.render_multiplane_comparison(
                organ_gt, correct, false, missed,
                output_path=output_path,
                title=f"{exp_name} - {organ_name}"
            )
        
        # Combined view
        combined_correct, combined_false, combined_missed = self.calculator.calculate_volume_comparison(
            pred_binary,
            (gt_multiclass > 0).astype(np.uint8)
        )
        
        output_path = output_dir / f"{exp_name}_All_Organs_multiplane.png"
        self.slice_renderer.render_multiplane_comparison(
            (gt_multiclass > 0).astype(np.uint8),
            combined_correct,
            combined_false,
            combined_missed,
            output_path=output_path,
            title=f"{exp_name} - All Organs"
        )
    
    def generate_per_organ_rotation(self, pred_volume, gt_volume, output_dir, exp_name):
        """Generate 3D rotation GIFs for each organ separately."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get unique organs from ground truth
        unique_organs = np.unique(gt_volume)
        unique_organs = unique_organs[unique_organs > 0]
        
        for organ_id in unique_organs:
            organ_name = ORGAN_NAMES.get(organ_id, f"Organ_{organ_id}")
            
            # Extract organ-specific masks
            organ_pred = (pred_volume == organ_id).astype(np.uint8)
            organ_gt = (gt_volume == organ_id).astype(np.uint8)
            
            # Calculate comparison
            correct, false, missed = self.calculator.calculate_volume_comparison(organ_pred, organ_gt)
            
            # Generate rotation
            output_path = output_dir / f"{exp_name}_{organ_name}_rotation_3d.gif"
            self.volume_renderer.render_comparison_rotation(
                organ_gt, correct, false, missed,
                output_path=output_path,
                title=f"{exp_name} - {organ_name}"
            )
        
        # Combined view
        combined_correct, combined_false, combined_missed = self.calculator.calculate_volume_comparison(
            (pred_volume > 0).astype(np.uint8),
            (gt_volume > 0).astype(np.uint8)
        )
        
        output_path = output_dir / f"{exp_name}_All_Organs_rotation_3d.gif"
        self.volume_renderer.render_comparison_rotation(
            (gt_volume > 0).astype(np.uint8),
            combined_correct,
            combined_false,
            combined_missed,
            output_path=output_path,
            title=f"{exp_name} - All Organs"
        )
    
    def generate_per_organ_rotation_binary(self, pred_binary, gt_multiclass, output_dir, exp_name):
        """Generate 3D rotation comparing binary prediction against each GT organ."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        unique_organs = np.unique(gt_multiclass)
        unique_organs = unique_organs[unique_organs > 0]
        
        for organ_id in unique_organs:
            organ_name = ORGAN_NAMES.get(organ_id, f"Organ_{organ_id}")
            organ_gt = (gt_multiclass == organ_id).astype(np.uint8)
            
            correct, false, missed = self.calculator.calculate_volume_comparison(pred_binary, organ_gt)
            
            output_path = output_dir / f"{exp_name}_{organ_name}_rotation_3d.gif"
            self.volume_renderer.render_comparison_rotation(
                organ_gt, correct, false, missed,
                output_path=output_path,
                title=f"{exp_name} - {organ_name}"
            )
        
        # Combined view
        combined_correct, combined_false, combined_missed = self.calculator.calculate_volume_comparison(
            pred_binary,
            (gt_multiclass > 0).astype(np.uint8)
        )
        
        output_path = output_dir / f"{exp_name}_All_Organs_rotation_3d.gif"
        self.volume_renderer.render_comparison_rotation(
            (gt_multiclass > 0).astype(np.uint8),
            combined_correct,
            combined_false,
            combined_missed,
            output_path=output_path,
            title=f"{exp_name} - All Organs"
        )
    
    def generate_per_organ_slices(self, pred_volume, gt_volume, output_dir, exp_name):
        """Generate slice animation GIFs for each organ separately."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get unique organs from ground truth
        unique_organs = np.unique(gt_volume)
        unique_organs = unique_organs[unique_organs > 0]
        
        for organ_id in unique_organs:
            organ_name = ORGAN_NAMES.get(organ_id, f"Organ_{organ_id}")
            
            # Extract organ-specific masks
            organ_pred = (pred_volume == organ_id).astype(np.uint8)
            organ_gt = (gt_volume == organ_id).astype(np.uint8)
            
            # Calculate comparison
            correct, false, missed = self.calculator.calculate_volume_comparison(organ_pred, organ_gt)
            
            # Generate slice animation
            output_path = output_dir / f"{exp_name}_{organ_name}_slices.gif"
            self.slice_renderer.render_slice_animation(
                organ_gt, correct, false, missed,
                output_path=output_path,
                title=f"{exp_name} - {organ_name}"
            )
        
        # Combined view
        combined_correct, combined_false, combined_missed = self.calculator.calculate_volume_comparison(
            (pred_volume > 0).astype(np.uint8),
            (gt_volume > 0).astype(np.uint8)
        )
        
        output_path = output_dir / f"{exp_name}_All_Organs_slices.gif"
        self.slice_renderer.render_slice_animation(
            (gt_volume > 0).astype(np.uint8),
            combined_correct,
            combined_false,
            combined_missed,
            output_path=output_path,
            title=f"{exp_name} - All Organs"
        )
    
    def generate_per_organ_slices_binary(self, pred_binary, gt_multiclass, output_dir, exp_name):
        """Generate slice animation comparing binary prediction against each GT organ."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        unique_organs = np.unique(gt_multiclass)
        unique_organs = unique_organs[unique_organs > 0]
        
        for organ_id in unique_organs:
            organ_name = ORGAN_NAMES.get(organ_id, f"Organ_{organ_id}")
            organ_gt = (gt_multiclass == organ_id).astype(np.uint8)
            
            correct, false, missed = self.calculator.calculate_volume_comparison(pred_binary, organ_gt)
            
            output_path = output_dir / f"{exp_name}_{organ_name}_slices.gif"
            self.slice_renderer.render_slice_animation(
                organ_gt, correct, false, missed,
                output_path=output_path,
                title=f"{exp_name} - {organ_name}"
            )
        
        # Combined view
        combined_correct, combined_false, combined_missed = self.calculator.calculate_volume_comparison(
            pred_binary,
            (gt_multiclass > 0).astype(np.uint8)
        )
        
        output_path = output_dir / f"{exp_name}_All_Organs_slices.gif"
        self.slice_renderer.render_slice_animation(
            (gt_multiclass > 0).astype(np.uint8),
            combined_correct,
            combined_false,
            combined_missed,
            output_path=output_path,
            title=f"{exp_name} - All Organs"
        )
    
    def process_experiment(self, phase: str, exp_name: str):
        """Process one experiment and generate all 3D visualizations by organ combination."""
        exp_path = self.viz_base / phase / exp_name
        output_dir = self.output_base / phase / exp_name
        
        print(f"\n{'='*80}")
        print(f"Processing: {phase}/{exp_name}")
        print(f"{'='*80}")
        
        # Load predictions from ALL splits (train, val, test)
        predictions = self.load_predictions_from_all_splits(exp_path)
        
        if not predictions:
            print("⚠️  No predictions found in any split")
            return False
        
        # Load ground truths (multiclass)
        gt_multiclass = self.load_multiclass_ground_truths(list(predictions.keys()))
        
        # Filter to matched pairs
        matched_ids = sorted(set(predictions.keys()) & set(gt_multiclass.keys()))
        
        if len(matched_ids) < 3:
            print(f"⚠️  Not enough matched images: {len(matched_ids)}")
            return False
        
        print(f"Found {len(matched_ids)} matched image pairs across all splits")
        
        # Group images by organ combination
        combination_groups = {}
        for combo_name, combo_images in self.organ_combinations.items():
            # Find which of our matched IDs belong to this combination
            combo_matched = [img_id for img_id in matched_ids if img_id in combo_images]
            if len(combo_matched) >= 3:  # Need at least 3 images for 3D
                combination_groups[combo_name] = combo_matched
                print(f"  {combo_name}: {len(combo_matched)} images")
        
        if not combination_groups:
            print("⚠️  No combinations with enough images (need ≥3 per combination)")
            return False
        
        # Process each organ combination separately
        for combo_name, combo_image_ids in combination_groups.items():
            print(f"\n  Processing combination: {combo_name}")
            
            # Create combined GT for this combination
            combo_gt = {}
            for img_id in combo_image_ids:
                gt_multiclass_mask = gt_multiclass[img_id]
                # Get classes that should be present in this combination
                target_classes = COMBINATION_CLASSES.get(combo_name, [])
                # Create binary mask: 1 if any target class present, 0 otherwise
                combined_mask = np.zeros_like(gt_multiclass_mask, dtype=np.uint8)
                for class_id in target_classes:
                    combined_mask |= (gt_multiclass_mask == class_id).astype(np.uint8)
                combo_gt[img_id] = combined_mask
            
            # Get predictions for this combination
            combo_pred = {img_id: predictions[img_id] for img_id in combo_image_ids}
            
            # Stack to volumes
            pred_volume = self.calculator.stack_slices_to_volume(combo_pred, combo_image_ids)
            gt_volume = self.calculator.stack_slices_to_volume(combo_gt, combo_image_ids)
            
            # Calculate comparison
            correct, false, missed = self.calculator.calculate_volume_comparison(pred_volume, gt_volume)
            
            # Create safe filename
            safe_combo_name = combo_name.replace('+', '_').replace(' ', '_')
            
            # Generate visualizations
            combo_output = output_dir / safe_combo_name
            
            print(f"    Generating multiplane...")
            multiplane_dir = combo_output / "multiplane"
            multiplane_dir.mkdir(parents=True, exist_ok=True)
            output_path = multiplane_dir / f"{exp_name}_{safe_combo_name}_multiplane.png"
            self.slice_renderer.render_multiplane_comparison(
                gt_volume, correct, false, missed,
                output_path=output_path,
                title=f"{exp_name} - {combo_name}"
            )
            
            print(f"    Generating 3D rotation...")
            rotation_dir = combo_output / "rotation_3d"
            rotation_dir.mkdir(parents=True, exist_ok=True)
            output_path = rotation_dir / f"{exp_name}_{safe_combo_name}_rotation_3d.gif"
            self.volume_renderer.render_comparison_rotation(
                gt_volume, correct, false, missed,
                output_path=output_path,
                title=f"{exp_name} - {combo_name}"
            )
            
            print(f"    Generating slice animation...")
            slices_dir = combo_output / "slices"
            slices_dir.mkdir(parents=True, exist_ok=True)
            output_path = slices_dir / f"{exp_name}_{safe_combo_name}_slices.gif"
            self.slice_renderer.render_slice_animation(
                gt_volume, correct, false, missed,
                output_path=output_path,
                title=f"{exp_name} - {combo_name}"
            )
        
        print(f"✓ Generated 3D visualizations for {phase}/{exp_name}")
        return True


def find_all_experiments(viz_base: Path):
    """Find all experiment folders."""
    experiments = []
    
    for phase_dir in sorted(viz_base.iterdir()):
        if not phase_dir.is_dir() or phase_dir.name.startswith('.'):
            continue
        
        for exp_dir in sorted(phase_dir.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
                continue
            
            # Check if test/predictions exists
            if (exp_dir / "test" / "predictions").exists():
                experiments.append({
                    'phase': phase_dir.name,
                    'name': exp_dir.name
                })
    
    return experiments


def main():
    parser = argparse.ArgumentParser(description='Generate 3D inference visualizations for optimization experiments')
    parser.add_argument('--viz_base', type=str, default='Optimization_Visualization',
                       help='Base directory containing 2D visualizations')
    parser.add_argument('--output_base', type=str, default='Optimization_3D_Project_Inference',
                       help='Output directory for 3D visualizations')
    parser.add_argument('--phase', type=str, help='Process specific phase only')
    parser.add_argument('--experiment', type=str, help='Process specific experiment only')
    args = parser.parse_args()
    
    viz_base = Path(args.viz_base)
    output_base = Path(args.output_base)
    
    if not viz_base.exists():
        print(f"❌ Visualization base not found: {viz_base}")
        return 1
    
    generator = Optimization3DInferenceGenerator(viz_base, output_base)
    
    # Find experiments to process
    if args.experiment and args.phase:
        experiments = [{'phase': args.phase, 'name': args.experiment}]
    elif args.phase:
        experiments = [e for e in find_all_experiments(viz_base) if e['phase'] == args.phase]
    else:
        experiments = find_all_experiments(viz_base)
    
    if not experiments:
        print("❌ No experiments found")
        return 1
    
    print("="*80)
    print(f"Optimization 3D Inference Generator")
    print(f"Found {len(experiments)} experiments")
    print("="*80)
    
    success_count = 0
    for exp in tqdm(experiments, desc="Processing experiments"):
        try:
            if generator.process_experiment(exp['phase'], exp['name']):
                success_count += 1
        except Exception as e:
            print(f"✗ Error processing {exp['phase']}/{exp['name']}: {str(e)}")
    
    print("\n" + "="*80)
    print(f"✅ Successfully processed {success_count}/{len(experiments)} experiments")
    print(f"Output: {output_base}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

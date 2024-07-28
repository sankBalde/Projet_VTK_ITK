"""
Auteur:
    Abdoulaye Baldé
    Majeur Image 2024
"""


import argparse
from src.utils import *

def main():
    parser = argparse.ArgumentParser(description="Suivi des changements d'une tumeur à partir de deux scans")

    parser.add_argument('--fixed_image_path', type=str, default='Data/case6_gre1.nrrd',
                        help="Chemin vers l'image fixe")
    parser.add_argument('--moving_image_path', type=str, default='Data/case6_gre2.nrrd',
                        help="Chemin vers l'image mobile")
    parser.add_argument('--registration_type', type=str, choices=['info_mutuelle'], default='info_mutuelle',
                        help="Type de recalage (par défaut : info_mutuelle)")
    parser.add_argument('--seed_x', type=int, default=120, help="Coordonnée x du point de départ pour la segmentation")
    parser.add_argument('--seed_y', type=int, default=80, help="Coordonnée y du point de départ pour la segmentation")
    parser.add_argument('--lower', type=int, default=0, help="Seuil inférieur pour la segmentation")
    parser.add_argument('--upper', type=int, default=255, help="Seuil supérieur pour la segmentation")

    args = parser.parse_args()

    fixed_image, registered_image_info_mutuelle = register_images_type(args.fixed_image_path,
                                                                       args.moving_image_path,
                                                                       type=args.registration_type)

    fixed_tumor = segment_tumor_v2(fixed_image, seedX=args.seed_x, seedY=args.seed_y, lower=args.lower, upper=args.upper)
    moving_tumor = segment_tumor_v2(registered_image_info_mutuelle, seedX=args.seed_x, seedY=args.seed_y, lower=args.lower, upper=args.upper)

    fixed_volume = calculate_volume(fixed_tumor)
    moving_volume = calculate_volume(moving_tumor)

    print(f"Fixed tumor volume: {fixed_volume} voxels")
    print(f"Moving tumor volume: {moving_volume} voxels")

    print(f"Différence de volume: {moving_volume - fixed_volume} voxels")
    intensity_difference = calculate_intensity_difference(fixed_tumor, moving_tumor)
    visualize_slices(itk.GetImageFromArray(intensity_difference))

if __name__ == "__main__":
    main()
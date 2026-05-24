"""Physically-grounded captions for the four standard HSI benchmarks.
Each class has a single canonical caption built from the three-clause
template (material category, characteristic absorption features,
typical reflectance regime).

The captions are used as the natural-language supervision signal during
contrastive pretraining and as the open-vocabulary prompts at inference.
"""

CAPTIONS = {
    "PaviaU": {
        1: ("Asphalt",
            "An asphalt road surface with strong hydrocarbon absorption near 1730 nm and uniformly low visible reflectance below 0.2."),
        2: ("Meadows",
            "A meadow of healthy short grass with a sharp chlorophyll red edge near 700 nm, strong near-infrared plateau, and weak water absorption around 1450 nm."),
        3: ("Gravel",
            "Loose construction gravel with weak silicate absorption around 2200 nm, moderate visible reflectance between 0.2 and 0.4, and a flat near-infrared spectrum."),
        4: ("Trees",
            "Deciduous tree canopy with broad chlorophyll absorption in red bands, near-infrared plateau above 0.5, and water absorption near 1450 nm."),
        5: ("Painted Metal Sheets",
            "Painted metal roofing sheets with flat visible reflectance above 0.4 and weak near-infrared spectral features."),
        6: ("Bare Soil",
            "Bare urban soil with steady increase from visible to short-wave infrared, weak iron-oxide absorption near 900 nm and clay absorption near 2200 nm."),
        7: ("Bitumen",
            "Bitumen waterproof sealing layer with very strong hydrocarbon absorption around 1730 nm and uniformly dark visible reflectance below 0.15."),
        8: ("Self-Blocking Bricks",
            "Self-blocking concrete pavement bricks with a carbonate overtone near 2330 nm and bright visible reflectance above 0.3."),
        9: ("Shadows",
            "Cast shadow region with uniformly low reflectance below 0.1 across all bands and no diagnostic absorption features."),
    },
    "PaviaC": {
        1: ("Water",
            "Open water body with high blue reflectance and monotonically decreasing reflectance through near-infrared, with strong water absorption above 900 nm."),
        2: ("Trees",
            "Tree canopy with chlorophyll red edge near 700 nm and strong near-infrared plateau above 0.4."),
        3: ("Asphalt",
            "Asphalt road surface with hydrocarbon absorption near 1730 nm and uniformly dark visible reflectance below 0.2."),
        4: ("Self-Blocking Bricks",
            "Self-blocking concrete pavement bricks with a carbonate overtone near 2330 nm and bright visible reflectance above 0.3."),
        5: ("Bitumen",
            "Bitumen sealing layer with strong hydrocarbon absorption around 1730 nm and very low visible reflectance below 0.15."),
        6: ("Tiles",
            "Clay roof tiles with high red reflectance, weak iron-oxide absorption near 870 nm, and moderate near-infrared rise."),
        7: ("Shadows",
            "Cast shadow region with uniformly low reflectance and no diagnostic absorption features."),
        8: ("Meadows",
            "Urban meadow grass with chlorophyll red edge near 700 nm and weak water absorption near 1450 nm."),
        9: ("Bare Soil",
            "Bare urban soil with weak iron-oxide absorption near 900 nm and gradual rise from visible to short-wave infrared."),
    },
    "IndianPines": {
        1: ("Alfalfa",
            "Alfalfa forage crop with sharp chlorophyll red edge near 700 nm, strong near-infrared plateau above 0.5, and water absorption around 1450 nm and 1940 nm."),
        2: ("Corn-notill",
            "Corn field without tillage residue, mixed soil-vegetation signature with chlorophyll red edge near 700 nm and moderate near-infrared plateau."),
        3: ("Corn-mintill",
            "Corn field with minimum tillage, soil-vegetation mixed pixels showing partial chlorophyll absorption and lower NIR than dense canopy."),
        4: ("Corn",
            "Mature corn canopy with strong chlorophyll absorption, high near-infrared plateau and pronounced water absorption near 1450 nm."),
        5: ("Grass-pasture",
            "Grass pasture with chlorophyll red edge near 700 nm, near-infrared plateau around 0.45, and weak water absorption near 1450 nm."),
        6: ("Grass-trees",
            "Mixed grass and tree cover with chlorophyll red edge near 700 nm, very high near-infrared plateau, and strong water absorption near 1940 nm."),
        7: ("Grass-pasture-mowed",
            "Mowed grass pasture with reduced chlorophyll content, weaker red edge near 700 nm, and lower near-infrared plateau than healthy pasture."),
        8: ("Hay-windrowed",
            "Cut hay rolled into windrows, showing dry-vegetation cellulose absorption near 2100 nm and reduced chlorophyll features."),
        9: ("Oats",
            "Oat crop with chlorophyll red edge near 700 nm, near-infrared plateau around 0.45, and water absorption near 1450 nm and 1940 nm."),
        10: ("Soybean-notill",
            "Soybean field without tillage, mixed soil-vegetation pixels with moderate chlorophyll red edge and partial NIR plateau."),
        11: ("Soybean-mintill",
            "Soybean field with minimum tillage, soil-vegetation mixed signature with weaker NIR than full canopy."),
        12: ("Soybean-clean",
            "Soybean field with clean cultivation, dense canopy with strong red edge near 700 nm and high near-infrared plateau."),
        13: ("Wheat",
            "Wheat crop with chlorophyll red edge near 700 nm and pronounced water absorption near 1450 nm and 1940 nm."),
        14: ("Woods",
            "Deciduous forest canopy with strong chlorophyll absorption, very high near-infrared plateau above 0.6, and broad water absorption."),
        15: ("Buildings-Grass-Trees-Drives",
            "Mixed urban surfaces including buildings, grass, trees, and driveways with heterogeneous reflectance and partial vegetation features."),
        16: ("Stone-Steel-Towers",
            "Stone and steel telecommunication towers with high broadband reflectance and weak iron-oxide absorption."),
    },
    "Salinas": {
        1: ("Brocoli_green_weeds_1",
            "Young broccoli field with green weeds, sharp chlorophyll red edge near 700 nm and high near-infrared plateau."),
        2: ("Brocoli_green_weeds_2",
            "Mature broccoli field with green weeds, similar chlorophyll signature to young broccoli but slightly higher near-infrared."),
        3: ("Fallow",
            "Fallow agricultural field with bare soil signature, steady increase from visible to short-wave infrared and weak iron-oxide absorption near 900 nm."),
        4: ("Fallow_rough_plow",
            "Fallow field after rough plowing, soil signature with shadow effects from clods producing lower overall reflectance."),
        5: ("Fallow_smooth",
            "Fallow field after smooth plowing, uniform bare soil signature with weak clay absorption near 2200 nm."),
        6: ("Stubble",
            "Crop stubble residue with dry-vegetation cellulose absorption near 2100 nm and reduced chlorophyll features."),
        7: ("Celery",
            "Celery crop canopy with chlorophyll red edge near 700 nm and high near-infrared plateau above 0.5."),
        8: ("Grapes_untrained",
            "Untrained grape vines with mixed soil-vegetation pixels and partial chlorophyll absorption."),
        9: ("Soil_vinyard_develop",
            "Vineyard soil during development stage with bare soil signature dominating over partial vine cover."),
        10: ("Corn_senesced_green_weeds",
            "Senescent corn field with green weeds, dry-vegetation cellulose absorption and reduced chlorophyll."),
        11: ("Lettuce_romaine_4wk",
            "Romaine lettuce at 4 weeks growth, early chlorophyll signature with low near-infrared plateau."),
        12: ("Lettuce_romaine_5wk",
            "Romaine lettuce at 5 weeks growth, stronger chlorophyll red edge and higher near-infrared plateau."),
        13: ("Lettuce_romaine_6wk",
            "Romaine lettuce at 6 weeks growth, mature chlorophyll signature with high near-infrared plateau."),
        14: ("Lettuce_romaine_7wk",
            "Romaine lettuce at 7 weeks growth, peak chlorophyll signature with very high near-infrared plateau."),
        15: ("Vinyard_untrained",
            "Untrained vineyard with mixed soil-vegetation pixels and partial chlorophyll absorption."),
        16: ("Vinyard_vertical_trellis",
            "Vineyard with vertical trellis system showing structured row pattern of grape canopy with chlorophyll red edge and bare soil between rows."),
    },
}


def get_caption(dataset: str, class_id: int) -> str:
    """Return the canonical caption for a (dataset, class_id) pair."""
    name, cap = CAPTIONS[dataset][class_id]
    return cap


def get_class_name(dataset: str, class_id: int) -> str:
    name, cap = CAPTIONS[dataset][class_id]
    return name


def all_prompts(dataset: str):
    """Return (class_ids, class_names, captions) lists for a dataset."""
    items = sorted(CAPTIONS[dataset].items())
    ids = [c for c, _ in items]
    names = [v[0] for _, v in items]
    caps = [v[1] for _, v in items]
    return ids, names, caps
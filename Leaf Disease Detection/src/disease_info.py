"""
Disease Information Database & Metadata Registry
Provides detailed domain knowledge for diagnosed leaf conditions.
"""

DISEASE_KNOWLEDGE_BASE = {
    "Tomato_Early_Blight": {
        "plant": "Tomato",
        "disease": "Early Blight (Alternaria solani)",
        "status": "Infected",
        "severity": "Moderate to High",
        "description": "Early Blight is a common fungal disease caused by Alternaria solani. It causes concentric dark ring spots on mature leaves, leading to defoliation and yield loss.",
        "symptoms": [
            "Concentric dark brown to black spots ('target-board' pattern) on older leaves.",
            "Yellow halo surrounding leaf lesions.",
            "Premature leaf drop starting from the bottom of the plant upward."
        ],
        "treatment": [
            "Apply copper-based fungicides or chlorothalonil at early symptom onset.",
            "Prune infected lower leaves to restrict fungal spore splash.",
            "Ensure proper crop spacing to improve air circulation."
        ],
        "prevention": [
            "Practice 2-3 year crop rotation with non-solanaceous crops.",
            "Use drip irrigation instead of overhead watering.",
            "Mulch soil around plant bases to prevent spore soil splash."
        ]
    },
    "Tomato_Late_Blight": {
        "plant": "Tomato",
        "disease": "Late Blight (Phytophthora infestans)",
        "status": "Infected",
        "severity": "Critical",
        "description": "Late Blight is a highly destructive oomycete pathogen capable of ruining entire crops within days during wet, cool conditions.",
        "symptoms": [
            "Large, dark, water-soaked lesions on leaf tips and margins.",
            "White fuzzy mold growth on the undersides of infected leaves during humid weather.",
            "Rapid stem browning and foliar collapse."
        ],
        "treatment": [
            "Apply systemic fungicides like Metalaxyl or Mancozeb immediately.",
            "Remove and incinerate infected plants to arrest spore dispersion.",
            "Avoid handling foliage when wet."
        ],
        "prevention": [
            "Plant certified disease-resistant tomato varieties.",
            "Monitor weather forecasts for high-humidity conditions.",
            "Destroy volunteer tomato and potato plants nearby."
        ]
    },
    "Tomato_Leaf_Spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot (Septoria lycopersici)",
        "status": "Infected",
        "severity": "Moderate",
        "description": "Septoria Leaf Spot is a fungal foliage infection characterized by numerous tiny circular lesions with gray centers.",
        "symptoms": [
            "Numerous small, circular spots (1-3 mm wide) with dark borders and tan/gray centers.",
            "Tiny black specks (pycnidia) inside lesion centers.",
            "Severe yellowing of affected leaves followed by premature defoliation."
        ],
        "treatment": [
            "Apply preventative bio-fungicides or copper sprays.",
            "Remove diseased lower leaves promptly.",
            "Sanitize garden tools after pruning infected foliage."
        ],
        "prevention": [
            "Maintain clean field borders free of weeds.",
            "Avoid overhead irrigation.",
            "Apply organic mulch to minimize soil splash onto lower foliage."
        ]
    },
    "Tomato_Powdery_Mildew": {
        "plant": "Tomato",
        "disease": "Powdery Mildew (Oidium neolycopersici)",
        "status": "Infected",
        "severity": "Mild to Moderate",
        "description": "Powdery Mildew appears as white flour-like powder on upper leaf surfaces, stunting photosynthesis and plant vigor.",
        "symptoms": [
            "White, talcum powder-like fungal patches on leaf surfaces.",
            "Leaf yellowing and curling under severe infection.",
            "Reduced fruit size due to compromised photosynthesis."
        ],
        "treatment": [
            "Spray neem oil, potassium bicarbonate, or sulfur-based organic fungicides.",
            "Apply bio-control agents such as Bacillus subtilis."
        ],
        "prevention": [
            "Provide ample sunlight and adequate plant spacing.",
            "Avoid excessive nitrogen fertilization."
        ]
    },
    "Tomato_Bacterial_Spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot (Xanthomonas spp.)",
        "status": "Infected",
        "severity": "High",
        "description": "Bacterial Spot affects leaves and fruit, causing small water-soaked spots that darken and dry out into dead tissue.",
        "symptoms": [
            "Small, angular, dark water-soaked spots on leaves.",
            "Leaf yellowing surrounding lesions and eventual leaf drop.",
            "Raised dark scab-like spots on green fruit."
        ],
        "treatment": [
            "Spray copper hydroxide mixed with mancozeb.",
            "Remove heavily infected plant debris from the field."
        ],
        "prevention": [
            "Use certified pathogen-free seeds and transplants.",
            "Avoid working in fields when foliage is wet."
        ]
    },
    "Tomato_Healthy": {
        "plant": "Tomato",
        "disease": "Healthy Leaf",
        "status": "Healthy",
        "severity": "None",
        "description": "The leaf exhibits vibrant green coloration, uniform leaf structure, and no visible signs of fungal or bacterial infection.",
        "symptoms": ["No disease symptoms detected. Leaf is in optimal health."],
        "treatment": ["No treatment required. Maintain current watering and nutrition routine."],
        "prevention": [
            "Continue regular agricultural monitoring.",
            "Ensure balanced fertilization and moisture levels."
        ]
    },
    "Potato_Early_Blight": {
        "plant": "Potato",
        "disease": "Early Blight (Alternaria solani)",
        "status": "Infected",
        "severity": "Moderate",
        "description": "Fungal infection affecting foliage and tubers, causing brown concentric ring spots on older potato leaves.",
        "symptoms": [
            "Brown concentric rings on mature foliage.",
            "Yellow foliage surrounding dark spots."
        ],
        "treatment": ["Apply protective fungicides (Chlorothalonil or Mancozeb)."],
        "prevention": ["Rotate crops with non-solanaceous species.", "Ensure balanced soil nitrogen levels."]
    },
    "Potato_Late_Blight": {
        "plant": "Potato",
        "disease": "Late Blight (Phytophthora infestans)",
        "status": "Infected",
        "severity": "Critical",
        "description": "Destructive blight that historically caused the Irish Potato Famine. Spreads rapidly in cool, moist weather.",
        "symptoms": [
            "Irregular water-soaked dark lesions.",
            "White mold growth on leaf undersides in humid conditions."
        ],
        "treatment": ["Destroy infected plants immediately.", "Apply systemic copper or metalaxyl fungicides."],
        "prevention": ["Plant certified disease-free seed potatoes.", "Avoid overhead irrigation."]
    },
    "Potato_Healthy": {
        "plant": "Potato",
        "disease": "Healthy Leaf",
        "status": "Healthy",
        "severity": "None",
        "description": "Healthy potato leaf with robust green foliage and standard leaf texture.",
        "symptoms": ["No disease symptoms present."],
        "treatment": ["Maintain routine crop management."],
        "prevention": ["Continue field monitoring and proper fertilization."]
    },
    "Apple_Scab": {
        "plant": "Apple",
        "disease": "Apple Scab (Venturia inaequalis)",
        "status": "Infected",
        "severity": "Moderate to High",
        "description": "Fungal disease causing olive-green to black velvet-like lesions on leaves and fruit, causing foliage deformation.",
        "symptoms": [
            "Olive-green to dark brown spots on upper leaf surfaces.",
            "Deformed, curled leaves that fall early in the season."
        ],
        "treatment": ["Apply captan or sulfur-based fungicides during bloom."],
        "prevention": ["Rake and destroy fallen leaves in autumn to remove overwintering spores."]
    },
    "Apple_Black_Rot": {
        "plant": "Apple",
        "disease": "Black Rot (Botryosphaeria obtusa)",
        "status": "Infected",
        "severity": "High",
        "description": "Causes 'frog-eye' leaf spots and dark rotting fruit on apple trees.",
        "symptoms": ["Purple spots that enlarge into brown lesions with dark borders ('frog-eye')."],
        "treatment": ["Prune dead or diseased wood during dormant winter months."],
        "prevention": ["Remove mummified fruit from trees and ground."]
    },
    "Apple_Healthy": {
        "plant": "Apple",
        "disease": "Healthy Leaf",
        "status": "Healthy",
        "severity": "None",
        "description": "Apple foliage displays normal deep green color with smooth leaf margins.",
        "symptoms": ["No lesions or leaf discoloration observed."],
        "treatment": ["No action required."],
        "prevention": ["Standard orchard maintenance and nutrient management."]
    },
    "Corn_Common_Rust": {
        "plant": "Corn (Maize)",
        "disease": "Common Rust (Puccinia sorghi)",
        "status": "Infected",
        "severity": "Moderate",
        "description": "Fungal disease marked by raised reddish-brown pustules on both upper and lower leaf surfaces.",
        "symptoms": ["Oval to elongate cinnamon-brown pustules that rupture leaf epidermis."],
        "treatment": ["Fungicide application if rust appears before tasseling on susceptible hybrids."],
        "prevention": ["Plant rust-resistant corn hybrids."]
    },
    "Corn_Healthy": {
        "plant": "Corn (Maize)",
        "disease": "Healthy Leaf",
        "status": "Healthy",
        "severity": "None",
        "description": "Healthy maize leaf showing uniform long green leaf blade with intact veins.",
        "symptoms": ["No rust pustules or chlorotic spots."],
        "treatment": ["Maintain field management."],
        "prevention": ["Ensure proper nitrogen fertilization."]
    }
}

DEFAULT_UNKNOWN_DISEASE = {
    "plant": "Plant Leaf",
    "disease": "Unspecified Leaf Condition",
    "status": "Unknown",
    "severity": "Undetermined",
    "description": "The leaf condition is analyzed based on computer vision patterns. Detailed domain metadata is pending laboratory verification.",
    "symptoms": ["Visual leaf features detected by deep learning model."],
    "treatment": ["Consult an agricultural expert or local agronomy extension service for physical inspection."],
    "prevention": ["Practice general crop sanitation and isolate affected plants."]
}

DISCLAIMER_TEXT = (
    "⚠️ **Agricultural & AI Disclaimer**: This deep learning diagnostic system is designed for decision support "
    "and educational inspection purposes. Predictions are generated via computer vision models and should not "
    "replace professional agronomic, laboratory, or botanical diagnostic confirmation."
)

def get_disease_details(class_name):
    """
    Retrieves disease information dictionary for a given class key.
    
    Args:
        class_name (str): Class string (e.g. 'Tomato_Early_Blight')
        
    Returns:
        dict: Information dictionary detailing plant, disease, status, symptoms, treatment, etc.
    """
    # Normalize class key formatting
    normalized_key = class_name.strip().replace(" ", "_")
    
    if normalized_key in DISEASE_KNOWLEDGE_BASE:
        return DISEASE_KNOWLEDGE_BASE[normalized_key]
    
    # Try case-insensitive matching
    for key, info in DISEASE_KNOWLEDGE_BASE.items():
        if key.lower() == normalized_key.lower():
            return info
            
    # Check if 'Healthy' is in the class name
    info = DEFAULT_UNKNOWN_DISEASE.copy()
    info["disease"] = class_name.replace("_", " ")
    if "healthy" in class_name.lower():
        info["status"] = "Healthy"
        info["severity"] = "None"
    else:
        info["status"] = "Infected"
        info["severity"] = "Moderate"
        
    return info

from fastai.vision.all import *
import gradio as gr
from PIL import Image
import numpy as np

# Load the learner
learn = load_learner('Cat_disease_resnet50_8.pkl')

# Define the categories and corresponding paragraphs
Categories = [
    'Cat with Fungal Infection',
    'Cat with Scabies',
    'Cat with Conjunctivitis (Pink Eye)',
    'Cat with Ear Mites',
    'Cat with Feline Acne',
    'Cat with Fleas'
]

Paragraphs = {
    'Cat with Fungal Infection': "Fungal infections in cats, such as ringworm, can be treated at home with careful management and hygiene practices. Start by isolating the infected cat to prevent the spread of the infection to other pets and humans. Clean the Environment: Thoroughly clean and disinfect areas where your cat spends time, including bedding, toys, and furniture. Use a solution of bleach and water (1:10 ratio) to kill fungal spores. Topical Treatments: Apply antifungal creams or ointments specifically designed for pets. Options like clotrimazole or miconazole can be effective. Always follow the instructions on the product label. Diet and Supplements: Consider adding antifungal supplements like probiotics to your cat's diet, which can help boost their immune system. Regular Grooming: Brush your cat regularly to remove dead hair and spores, but be cautious not to irritate the skin. If symptoms persist or worsen, consult a veterinarian for prescription antifungal medications.",
    'Cat with Scabies': "Scabies in cats, caused by the Notoedris cati mite, requires immediate attention. Home treatment should focus on reducing discomfort and preventing the spread of mites. Isolation: Keep the affected cat away from other pets to prevent transmission. Lime Sulfur Dip: This is an effective over-the-counter treatment. Dilute lime sulfur with water and apply it to your cat's skin, ensuring it remains on for the recommended duration without rinsing off. A cone collar may be necessary to prevent licking. Clean Bedding and Toys: Wash all bedding, toys, and any items the cat frequently contacts in hot water. Discard items that cannot be washed. Environmental Management: Vacuum and clean your home thoroughly to remove any mites that may have fallen off your cat. Monitor your cat closely and consult a veterinarian if the condition does not improve.",
    'Cat with Conjunctivitis (Pink Eye)': "Conjunctivitis in cats can be caused by allergies, infections, or irritants. Home care can help alleviate symptoms while you seek veterinary advice. Warm Compresses: Apply a warm, damp cloth to your cat's eyes for a few minutes several times a day. This can help soothe irritation and remove discharge. Saline Solution: Use a sterile saline solution to gently flush your cat's eyes. This can help remove debris and soothe irritation. Avoid Irritants: Ensure your cat is in a clean environment free from dust, smoke, and strong odors that could exacerbate the condition. Monitor Symptoms: Keep an eye on your cat’s symptoms. If redness, swelling, or discharge persists, consult a veterinarian for appropriate medications.",
    'Cat with Ear Mites': "Ear mites are common parasites that can cause significant discomfort for cats. Home treatment should focus on cleaning the ears and using natural remedies. Ear Cleaning: Use a gentle ear cleaner or a mixture of equal parts apple cider vinegar and water to clean your cat's ears. Apply a few drops, massage the base of the ear, and wipe away debris with a cotton ball. Olive Oil: Apply a few drops of warm olive oil to your cat's ears. This can suffocate the mites and help in cleaning the ears when wiped out gently. Monitor for Secondary Infections: Keep an eye out for signs of infection, such as foul odor or excessive redness. If these occur, consult a veterinarian. Preventive Measures: Regularly check and clean your cat’s ears to prevent future infestations.",
    'Cat with Feline Acne': "Feline acne is a common condition characterized by blackheads and pimples, primarily on the chin. Home treatment focuses on hygiene and management. Change Food and Water Bowls: Switch to stainless steel or ceramic bowls, as plastic can harbor bacteria that worsen acne. Regular Cleaning: Clean your cat's chin with a damp cloth daily to remove debris and oil. You can also use cat-safe wipes designed for skin cleaning. Topical Treatments: Consult your veterinarian about using benzoyl peroxide wipes specifically formulated for cats to help reduce acne. Grooming: Regularly groom your cat to prevent hair from trapping oils and dirt, which can exacerbate the condition. If the acne does not improve or worsens, seek veterinary advice for potential antibiotic treatments.",
    'Cat with Fleas': "Flea infestations can be distressing for both cats and their owners. Effective home management is crucial to eliminate fleas and prevent their return. Vacuum Thoroughly: Regularly vacuum carpets, furniture, and any areas your cat frequents. Dispose of the vacuum bag or empty the canister outside to prevent fleas from re-entering your home. Wash Bedding: Wash your cat's bedding and any fabric items in hot water to kill fleas and their eggs. Flea Comb: Use a fine-toothed flea comb to remove fleas from your cat's fur. Dip the comb in soapy water to kill any fleas you catch. Natural Remedies: Consider using natural flea repellents like diatomaceous earth (ensure it’s food-grade) sprinkled in areas where your cat spends time. Monitor and Treat: Keep an eye on your cat for signs of fleas and consider using vet-recommended flea treatments if the infestation persists. If these home remedies do not resolve the flea problem, consult a veterinarian for more effective treatments."
}

def classify_image(img):
    # Convert to PIL Image if not already
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.resize((192, 192))  # Resize the image to the required size
    pred, idx, prob = learn.predict(img)  # Predict using the model

    # Ensure prob is a numpy array
    prob = np.array(prob)

    # Get the top 2 categories with highest probabilities
    top_2_idx = prob.argsort()[-2:][::-1]  # Sort indices and select top 2
    top_2_categories = [Categories[i] for i in top_2_idx]
    top_2_paragraphs = [Paragraphs[category] for category in top_2_categories]

    # Prepare the output
    label_output = {Categories[i]: float(prob[i]) for i in range(len(Categories))}
    paragraphs_output = "\n\n".join(top_2_paragraphs)

    return label_output, paragraphs_output

# Setup Gradio interface
image = gr.Image()
label = gr.Label(label="Class Probabilities")
paragraphs = gr.Textbox(lines=10, label="Preliminary Treatment info on Top Two Prediction")
examples = [["Fleas2.png"], ["felan acne1.png"], ["PinkEye2.png"], ["Scabbies_1.png"],["Fungal_infection1.png"]]

intf = gr.Interface(fn=classify_image, inputs=image, outputs=[label, paragraphs], examples=examples, flagging_dir="/tmp/flagged")
intf.launch(inline=False)

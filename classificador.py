import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
from collections import defaultdict

# Dicionário de animais com possíveis sinônimos e variações
ANIMAL_NAMES = {
    "alpaca": ["alpaca"],
    "anta": ["tapir", "anta"],
    "aranha": ["spider", "aranha"],
    "babuino": ["baboon", "babuino"],
    "baleia azul": ["blue whale", "baleia azul"],
    "bisao americano": ["american bison", "bisao americano"],
    "borboleta": ["butterfly", "borboleta"],
    "bufalo dagua": ["water buffalo", "bufalo dagua"],
    "cabra montesa": ["mountain goat", "cabra montesa"],
    "cachorro": ["dog", "cachorro", "canine"],
    "camelo": ["camel", "camelo"],
    "canguru": ["kangaroo", "canguru"],
    "capivara": ["capybara", "capivara"],
    "carneiro": ["ram", "sheep", "carneiro"],
    "cavalo": ["horse", "cavalo"],
    "chacal": ["jackal", "chacal"],
    "coala": ["koala", "coala"],
    "doninha": ["weasel", "doninha"],
    "elefante": ["elephant", "elefante"],
    "esquilo": ["squirrel", "esquilo"],
    "esquilo planador": ["flying squirrel", "esquilo planador"],
    "foca": ["seal", "foca"],
    "gado das highlands": ["highland cattle", "gado das highlands"],
    "galinha": ["chicken", "galinha"],
    "gamba": ["shrimp", "gamba"],
    "gato": ["cat", "gato", "feline"],
    "girafa": ["giraffe", "girafa"],
    "gnu": ["wildebeest", "gnu"],
    "golfinho": ["dolphin", "golfinho"],
    "iaque": ["yak", "iaque"],
    "javali": ["wild boar", "javali"],
    "leao marinho": ["sea lion", "leao marinho"],
    "leopardo das neves": ["snow leopard", "leopardo das neves"],
    "lobo guara": ["maned wolf", "lobo guara"],
    "lontra": ["otter", "lontra"],
    "mangusta": ["mongoose", "mangusta"],
    "marmota": ["marmot", "marmota"],
    "morcego vampiro": ["vampire bat", "morcego vampiro"],
    "morsa": ["walrus", "morsa"],
    "onca pintada": ["jaguar", "onca pintada"],
    "orangotango": ["orangutan", "orangotango"],
    "panda vermelho": ["red panda", "panda vermelho"],
    "peixe boi": ["manatee", "peixe boi"],
    "porco espinho": ["porcupine", "porco espinho"],
    "raposa artica": ["arctic fox", "raposa artica"],
    "rinoceronte": ["rhinoceros", "rinoceronte"],
    "tamandua": ["anteater", "tamandua"],
    "tatu": ["armadillo", "tatu"],
    "texugo": ["badger", "texugo"],
    "urso pardo": ["brown bear", "urso pardo"],
    "urso polar": ["polar bear", "urso polar"],
    "vaca": ["cow", "vaca"],
    "vicunha": ["vicuña", "vicunha"],
    "vombate": ["wombat", "vombate"],
    "zebra": ["zebra"]
}

# Baixa classes do ImageNet
response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
imagenet_classes = [line.strip().lower() for line in response.text.splitlines()]

# Modelo e transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.mobilenet_v2(weights="DEFAULT")
model.eval()

def classify_animal(image: Image.Image):
    try:
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image)
        
        # Pega as top 3 predições
        _, top3 = outputs.topk(3)
        top3 = top3.squeeze().tolist()
        
        # Verifica cada predição contra nossa lista
        for idx in top3:
            predicted_label = imagenet_classes[idx].lower()
            
            # Busca direta por correspondência
            for animal in TARGET_ANIMALS:
                if animal in predicted_label or any(syn in predicted_label for syn in ANIMAL_NAMES.get(animal, [])):
                    return animal  # Retorna o nome formatado do nosso dicionário
        
        return "animal não identificado"  # Padrão mais descritivo
    
    except Exception as e:
        print(f"Erro na classificação: {e}")
        return "animal"
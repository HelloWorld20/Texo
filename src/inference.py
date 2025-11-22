from texo.data.processor import EvalMERImageProcessor
from texo.model.formulanet import FormulaNet
from transformers import AutoTokenizer, VisionEncoderDecoderModel, PreTrainedTokenizerFast
from PIL import Image
import torch


def load(path):
    model = VisionEncoderDecoderModel.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

def inference(model: FormulaNet, image_path: str, tokenizer: PreTrainedTokenizerFast, device):
    model.to(device)
    image = Image.open(image_path)
    image_processor = EvalMERImageProcessor(image_size={'width':384, 'height':384})
    processed_image = image_processor(image).unsqueeze(0)
    outputs = model.generate(pixel_values=processed_image.to(device))
    pred_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return pred_str


model, tokenizer = load('../model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_path = '../TechnoSelection/test_img/单行公式.png'
# display(IPython.display.Image(image_path))
pred_str = inference(model, image_path, tokenizer, device)
print(pred_str)
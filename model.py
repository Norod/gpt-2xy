import random
import re
import torch
from torch.nn import functional as F
from pytorch_transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
torch.set_grad_enabled(False)

MODEL_PATH = './WoWQuestPytorch124M'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).eval()
model = model.to(device)

wow_class_list = ['Death Knight', 'Demon Hunter', 'Druid', 'Hunter', 'Mage', 'Monk', 'Paladin', 'Priest', 'Rogue', 'Shaman', 'Warrior', 'Warlock']
wow_race_list = ['Blood Elf', 'Human', 'Tauren', 'Orc', 'Kul Tiran', 'Void Elf', 'Troll', 'Vulpera', 'Night Elf', 'Zandalari Troll', 'Worgen', 'Undead', 'Goblin', 'Highmountain Tauren', 'Nightborne', 'Dwarf', 'Draenei', 'Gnome', 'Lightforged Draenei', 'Pandaren', 'Maghar Orc', 'Mechagnome', 'Dark Iron Dwarf']
wow_silly_name_list = ['Glitterstorm', 'Sunderwear', 'Arrowdynamic', 'Sapntap', 'Crossblesser', 'Praystation', 'Healium', 'Shocknorris', 'Alestrom', 'Harryportal', 'Merlìn', 'Wreckquiem', 'Owlcapone']

suggested_text_list = ['Greetings $r', '$c I need your help', 'Good to see you $n']

def parseGenderTokens(text):
    regex = r"\$[gG]([^:]+):([^;]+);"
    matches = re.finditer(regex, text, re.MULTILINE)
    parsed_string = ""
    prev_index = 0
    group_num = 0
    random_group = -1    
    for matchNum, match in enumerate(matches, start=1):    
        parsed_string += text[prev_index:match.start()]            
        if random_group == -1:
            group_num = len(match.groups())
            random_group = random.randint(1, group_num)
        parsed_string += match.group(random_group)
        prev_index = match.end(group_num) + 1
    parsed_string += text[prev_index:]
    return parsed_string

def parseSpecialCharacters(text, wow_class_item, wow_race_item, wow_silly_name_item):
    parsedText = text.replace("$B", "\n").replace("$b", "\n").replace("$c", wow_class_item).replace("$C", wow_class_item).replace("$r", wow_race_item).replace("$R", wow_race_item).replace("$n", wow_silly_name_item).replace("$N", wow_silly_name_item)
    return parseGenderTokens(parsedText)

def extend(text, size=60):
    if len(text) == 0:
        text = random.choice(suggested_text_list)
    tokens = tokenizer.encode(text)
    prediction, past = torch.tensor([tokens]).to(device), None
    for i in range(size):
        prediction, past = model(prediction, past=past)
        prediction = torch.multinomial(F.softmax(prediction[:, -1], dim=1), 1)
        tokens.append(prediction.item())
    decoded_tokens = tokenizer.decode(tokens)
    wow_class_item = random.choice(wow_class_list)
    wow_race_item = random.choice(wow_race_list)
    wow_silly_name_item = random.choice(wow_silly_name_list)
    return parseSpecialCharacters(decoded_tokens, wow_class_item, wow_race_item, wow_silly_name_item)


if __name__ == "__main__":
    random.seed(None)
    #test_text = '$c, over here. Hello $n the $r I need your help'
    #test_text = "Hello there $gGentelman:Lady;, how are you $gBoy:Girl;?"
    test_text = 'I need your help'
    extended = extend(test_text, 120)
    print(extended)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertForSequenceClassification, AutoTokenizer

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)

from sklearn.calibration import calibration_curve
from tqdm import tqdm

import time

from data.dataset import IMDBDataset
from configs.get_config import get_config
from models.lstm_model import LSTMClassifier

test_cases = [
    ["This movie is absolutely fantastic! Best film I've seen all year.", 1],
    ["I loved every minute of it. Highly recommend!", 1],
    ["Brilliant acting, amazing story, beautiful cinematography. A masterpiece!", 1],
    ["Mind-blowing! One of the best movies ever made.", 1],
    ["Perfect film from start to finish. Can't find a single flaw.", 1],
    ["Absolutely incredible! I'm speechless.", 1],
    ["A must-watch for everyone. Pure genius.", 1],
    ["Touching, emotional, and brilliant. 10/10.", 1],
    ["I couldn't stop smiling throughout the entire film.", 1],
    ["Exceeded all my expectations. What a ride!", 1],
    
    # Positive with minor flaws
    ["Really enjoyed it. A few slow moments but overall great.", 1],
    ["Not perfect but definitely worth watching. Solid entertainment.", 1],
    ["Great characters, good plot, enjoyable experience.", 1],
    ["Better than I expected. Would watch again.", 1],
    ["A bit predictable but still fun and engaging.", 1],
    ["The soundtrack alone makes this worth seeing.", 1],
    ["Strong performances carry this film.", 1],
    ["Heartwarming and funny. A good time at the movies.", 1],
    ["Visually stunning with a compelling story.", 1],
    ["I had a blast watching this.", 1],

    ["Oh wow, another generic superhero movie. How original.", 0],
    ["Yeah, because what the world needed was another pointless sequel. Brilliant.", 0],
    ["The acting was so convincing, I almost believed they weren't reading cue cards.", 0],
    ["I love paying money to watch paint dry for two hours. Money well spent.", 0],
    ["This film really raised the bar... for how low a movie can go.", 0],
    
    # Скрытый сарказм через преувеличение
    ["Congratulations to the director for making a film that puts insomniacs to sleep instantly.", 0],
    ["The plot twists were so unexpected that I predicted them all in the first 10 minutes.", 0],
    ["I've never seen such creative use of recycled Hollywood clichés. Truly groundbreaking.", 0],
    ["This movie deserves an Oscar... for the worst screenplay of the decade.", 0],
    ["The special effects were so realistic that I could clearly see the green screen.", 0],
    
    # Сравнительный сарказм
    ["This film makes 'The Room' look like Citizen Kane. And that's saying something.", 0],
    ["If you thought the first movie was bad, wait until you suffer through this masterpiece of mediocrity.", 0],
    ["The director must have learned filmmaking from YouTube tutorials. And skipped the good ones.", 0],
    
    # Сарказм с "позитивной" лексикой
    ["What a delightfully boring experience from start to finish.", 0],
    ["I'm so grateful that this movie wasted 2 hours of my life that I'll never get back.", 0],
    ["The ending was wonderfully predictable. I couldn't have been more surprised.", 0],
    ["Such a refreshing take on storytelling – if by refreshing you mean utterly derivative.", 0],
    
    # Ироничные вопросы
    ["Who needs character development when you can have explosions, right?", 0],
    ["Why would anyone want a coherent plot when we can just watch the actors looking confused?", 0],
    ["What's better than a movie that insults your intelligence at every turn? Absolutely nothing.", 0],
    
    # Длинный саркастичный отзыв
    ["I have to thank this movie for teaching me something valuable – how to fall asleep with my eyes open. The 'twist' was so obvious that my cat figured it out. The lead actor delivered every line with the emotional range of a cardboard box.", 0],

    # Positive - short/emojis
    ["Loved it!", 1],
    ["Best movie ever! 🔥", 1],
    ["Incredible!!!", 1],
    ["Wow. Just wow.", 1],
    ["10/10 would recommend", 1],
    
    # ========== NEGATIVE (0) ==========
    # Clear Negative
    ["Terrible movie. Waste of time and money. Avoid at all costs!", 0],
    ["One of the worst films I've ever seen. Don't bother.", 0],
    ["Awful acting, boring plot, and horrible ending. Complete disappointment.", 0],
    ["Complete garbage. I want my money back.", 0],
    ["Unwatchable. Turned it off after 20 minutes.", 0],
    ["Painfully bad. Everything about this film is wrong.", 0],
    ["A disaster from start to finish.", 0],
    ["Horrible script, wooden acting, zero chemistry.", 0],
    ["I'd rather watch paint dry. So boring.", 0],
    ["Insultingly stupid. Waste of talent and resources.", 0],
    
    # Negative with minor positives
    ["Good visuals but terrible story. Overall disappointing.", 0],
    ["The acting was great but the story was boring. Not sure what to think.", 0],
    ["Started strong but fell apart in the third act. Disappointing ending.", 0],
    ["Beautiful cinematography wasted on a pointless plot.", 0],
    ["Great premise but terrible execution.", 0],
    ["Some funny moments but mostly boring.", 0],
    ["Good effort but fails to deliver.", 0],
    ["The actors tried their best but the script let them down.", 0],
    ["Impressive visuals, empty soul.", 0],
    ["Promising start, disappointing finish.", 0],
    
    # Negative - short
    ["Waste of time", 0],
    ["Complete trash", 0],
    ["Don't bother watching", 0],
    ["Terrible from start to finish", 0],
    ["Worst movie ever", 0],
    
    # ========== SARCASIS/COMPLEX (0 = негативный) ==========
    ["Oh great, another superhero movie. Just what we needed.", 0],
    ["I've seen worse, but I've definitely seen much better.", 0],
    ["This film is so bad it's actually kind of good?", 0],
    ["Thanks for wasting 2 hours of my life. Really appreciate it.", 0],
    ["Yeah, because we definitely needed another sequel.", 0],
    ["Brilliant. Just brilliant. (sarcasm)", 0],
    ["What a masterpiece... of garbage.", 0],
    ["This film really raised the bar... for disappointment.", 0],
    ["Exactly what you'd expect from a movie that took 2 weeks to write.", 0],
    ["Sure, it's 'entertaining' if you turn your brain off completely.", 0],
    
    # ========== NEUTRAL/MIXED (0 - склоняемся к негативному) ==========
    ["It's okay. Not terrible, but not great either. Average movie.", 0],
    ["Meh. It exists. That's about all I can say.", 0],
    ["I have no strong feelings about this film.", 0],
    ["It was alright I guess. Nothing special though.", 0],
    ["Decent way to kill 2 hours but forgettable.", 0],
    ["Not bad, not good. Just there.", 0],
    ["Could be better, could be worse.", 0],
    ["It passed the time. That's all.", 0],
    ["Some good parts, some bad parts. Overall meh.", 0],
    ["A solid 5/10. Nothing more, nothing less.", 0],
    
    # ========== ADDITIONAL POSITIVE ==========
    ["What a beautiful film! Left me in tears.", 1],
    ["Absolutely wonderful. Can't recommend enough.", 1],
    ["Pure joy from beginning to end.", 1],
    ["A hidden gem. So glad I watched this.", 1],
    ["Powerful, moving, unforgettable.", 1],
    ["The best film of the year and it's not even close.", 1],
    ["Flawless execution. Every scene works.", 1],
    ["I'm buying this on Blu-ray as soon as it comes out.", 1],
    ["Finally, a movie that respects its audience.", 1],
    ["This film changed how I see the world.", 1],
    
    # ========== ADDITIONAL NEGATIVE ==========
    ["What were they thinking? This is unwatchable.", 0],
    ["A complete mess from every angle.", 0],
    ["Save your money. Seriously.", 0],
    ["I want those 2 hours of my life back.", 0],
    ["Embarrassingly bad.", 0],
    ["The trailer was the best part of this movie.", 0],
    ["How did this get funded?", 0],
    ["My expectations were low but holy crap.", 0],
    ["Painfully predictable and boring.", 0],
    ["This film has no redeeming qualities.", 0],

    # Прямые позитивные (простые случаи)
    ["This movie was absolutely fantastic! I loved every minute of it.", 1],
    ["Brilliant acting, stunning visuals, and a gripping story. A masterpiece!", 1],
    ["One of the best films I've seen this year. Highly recommend!", 1],
    ["The plot kept me on the edge of my seat from start to finish.", 1],
    ["Amazing soundtrack and incredible performances by the entire cast.", 1],
    
    # С умеренными усилителями (легко, но разнообразнее)
    ["Pretty good movie overall, enjoyed watching it with my family.", 1],
    ["Solid 8/10. Great character development and unexpected twists.", 1],
    ["Really nice cinematography, definitely worth watching.", 1],
    ["Not bad at all, actually quite entertaining!", 1],
    ["Surprisingly good! I didn't expect much but was pleasantly surprised.", 1],
    
    # С идиомами и устойчивыми выражениями (средняя сложность)
    ["This film hits all the right notes. Pure gold from beginning to end.", 1],
    ["A hidden gem that deserves way more attention than it got.", 1],
    ["The director knocked it out of the park with this one.", 1],
    ["It's a feel-good movie that leaves you smiling long after the credits roll.", 1],
    
    # С отсылками к жанрам/элементам (требуют понимания контекста)
    ["If you're a fan of sci-fi, this is a must-watch. The world-building is top-notch.", 1],
    ["A perfect blend of humor and heart. The comedic timing is flawless.", 1],
    ["The chemistry between the leads is undeniable and carries the whole film.", 1],
    
    # С сарказмом или отрицательной лексикой, но позитивным посылом (сложные случаи для LSTM)
    ["I can't believe how good this movie is! It's crazy that I almost skipped it.", 1],
    ["There's nothing bad to say about this film. It's simply wonderful.", 1],
    ["Who knew a movie about paint drying could be so incredibly fascinating!", 1],
    ["Don't listen to the haters, this film is actually brilliant and thought-provoking.", 1],
    
    # Длинный отзыв с несколькими предложениями (проверка на длинный контекст)
    ["I came into this movie with zero expectations, but honestly, it blew me away. ", 1],
    ["The acting was raw and emotional, the director took real risks, and the ending was perfect. ", 1],
    ["It's been three days since I watched it and I can't stop thinking about it. Highly recommended!", 1],
]

def  test_lstm(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg['transformer_name'])
    max_len = cfg['max_len']
    batch_size = 16
    device = 'cuda'

    texts = [i[0] for i in test_cases]
    labels = [i[1] for i in test_cases]

    test_dataset = IMDBDataset(texts, labels, tokenizer, max_len=max_len)    
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    all_preds = []
    all_labels = []
    all_probs = []
    
    start_time = time.time()

    model = LSTMClassifier(tokenizer.vocab_size, cfg).to(device)
    model.load_state_dict(torch.load(cfg['models_save_path'] + '/lstm_model_20.pt')['model_state_dict'])

    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evulating lstm"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    inference_time = time.time() - start_time


    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    positive_probs = [p[1] for p in all_probs]

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Inference: {inference_time:.2f} sec")

def test_bert(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg['transformer_name'])
    max_len = cfg['max_len']
    batch_size = 12
    device = 'cuda'

    texts = [i[0] for i in test_cases]
    labels = [i[1] for i in test_cases]

    test_dataset = IMDBDataset(texts, labels, tokenizer, max_len=max_len)    
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    all_preds = []
    all_labels = []
    all_probs = []
    
    start_time = time.time()

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=cfg['num_classes']
    )

    model.load_state_dict(torch.load(cfg['models_save_path'] + '/bert_model_4.pt')['model_state_dict'])

    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evulating bert"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)

            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    inference_time = time.time() - start_time


    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    positive_probs = [p[1] for p in all_probs]

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Inference: {inference_time:.2f} sec")

def compare_models(config_path):
    cfg = get_config(config_path)

    test_lstm(cfg)
    test_bert(cfg)



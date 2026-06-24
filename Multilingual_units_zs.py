import os
import pandas as pd
import torch
import re
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from deep_translator import GoogleTranslator
from IPython.display import display
from transformers import BitsAndBytesConfig

# =========================
# 🔹 Translation & Cleaning helpers
# =========================

def clean_text(text):
    text = text.replace("#", " ")
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def translate(text, target_lang):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        return text

def translate_to_english(text):
    cleaned = clean_text(text)
    try:
        return GoogleTranslator(source='auto', target='en').translate(cleaned)
    except:
        return cleaned

# =========================
# 🔹 Language Groups & Config
# =========================

LANG_GROUPS = {
    "high": ["en", "zh"],
    "medium": ["el", "tr"],
    "low": ["sw", "ka"]
}

LANG_CONFIG = {
    "en": {"answer": "Answer the following question:", "instruction": "DO NOT add any explanation...", "return_line": "Return ONLY one line.", "format": "OUTPUT FORMAT:", "final": "The final answer is: <number>"},
    "zh": {"answer": "请回答以下问题：", "instruction": "不要添加任何解释或评论。", "return_line": "只返回一行。", "format": "输出格式：", "final": "最终答案是： <一个数字>"},
    "el": {"answer": "Απαντήστε στην ακόλουθη ερώτηση:", "instruction": "ΜΗΝ προσθέσετε καμία εξήγηση...", "return_line": "Επιστρέψτε ΜΟΝΟ μία γραμμή.", "format": "ΜΟΡΦΗ ΕΞΟΔΟΥ:", "final": "Η τελική απάντηση είναι: <ένας αριθμός>"},
    "tr": {"answer": "Aşağıdaki soruyu cevaplayın:", "instruction": "Herhangi bir açıklama...", "return_line": "SADECE tek satır döndürün.", "format": "ÇIKTI FORMATI:", "final": "Son cevap: <bir sayı>"},
    "sw": {"answer": "Jibu swali lifuatalo:", "instruction": "Usiongeze maelezo...", "return_line": "Rudisha mstari mmoja TU.", "format": "MUUNDO WA MATOKEO:", "final": "Jibu la mwisho ni: <nambari>"},
    "ka": {"answer": "უპასუხეთ შემდეგ კითხვას:", "instruction": "არ დაამატოთ ახსნა...", "return_line": "დააბრუნეთ მხოლოდ ერთი ხაზი.", "format": "გამოტანის ფორმატი:", "final": "საბოლოო პასუხია: <რიცხვი>"}
}

WORLD_CONFIG = {
    "en": "The value of {constant} is {value}",
    "el": "Η τιμή του {constant} είναι {value}",
    "zh": "{constant} 的值是 {value}",
    "tr": "{constant} değerinin değeri {value}",
    "sw": "Thamani ya {constant} ni {value}",
    "ka": "{constant}-ის მნიშვნელობა არის {value}"
}
QUESTION_CONFIG={}
QUESTION_CONFIG["level1"] = {
    1: {"en": "How many seconds are in 2 minutes?", "el": "Πόσα δευτερόλεπτα υπάρχουν σε 2 λεπτά;", "zh": "2 分钟有多少秒？", "tr": "2 dakikada kaç saniye vardır?", "sw": "Kuna sekunde ngapi katika dakika 2?", "ka": "რამდენი წამია 2 წუთში?"},
    2: {"en": "How many grams are in 2 kilograms?", "el": "Πόσα γραμμάρια υπάρχουν σε 2 κιλά;", "zh": "2 千克有多少克？", "tr": "2 kilogramda kaç gram vardır?", "sw": "Kuna gramu ngapi katika kilo 2?", "ka": "რამდენი გრამია 2 კილოგრამში?"},
    3: {"en": "How many centimeters are in 2 meters?", "el": "Πόσα εκατοστά υπάρχουν σε 2 μέτρα;", "zh": "2 米有多少厘米？", "tr": "2 metrede kaç santimetre vardır?", "sw": "Kuna sentimita ngapi katika mita 2?", "ka": "რამდენი სანტიმეტრია 2 მეტრში?"},
    4: {"en": "What is the Kelvin temperature when it is 0°C?", "el": "Ποια είναι η θερμοκρασία σε Kelvin όταν είναι 0°C;", "zh": "0°C 时的开尔文温度是多少？", "tr": "0°C sıcaklıkta Kelvin değeri nedir?", "sw": "Joto la Kelvin ni gani wakati ni 0°C?", "ka": "რა არის ტემპერატურა კელვინში როცა არის 0°C?"},
    5: {"en": "How many milliliters are in 1 cubic centimeter?", "el": "Πόσα ml υπάρχουν σε 1 κυβικό εκατοστό;", "zh": "1 立方厘米等于多少毫升？", "tr": "1 santimetreküpte kaç mililitre vardır?", "sw": "Kuna mililita ngapi katika sentimita ya ujazo 1?", "ka": "რამდენი მილილიტრია 1 კუბურ სანტიმეტრში?"},
    6: {"en": "How many Joules are in 3 calories?", "el": "Πόσα Joule είναι 3 θερμίδες;", "zh": "3 卡路里是多少焦耳？", "tr": "3 kalori kaç Joule eder?", "sw": "Kalori 3 ni Joule ngapi?", "ka": "3 კალორია რამდენ ჯოულს უდრის?"},
    7: {"en": "How many Pascals are in 2 atmospheres?", "el": "Πόσα Pascal είναι 2 ατμόσφαιρες;", "zh": "2 个大气压是多少帕斯卡？", "tr": "2 atmosfer kaç Pascal eder?", "sw": "Atmosfera 2 ni Pascal ngapi?", "ka": "2 ატმოსფერო რამდენ პასკალს უდრის?"},
    8: {"en": "How many millivolts are in 5 volts?", "el": "Πόσα millivolt είναι 5 volt;", "zh": "5 伏特是多少毫伏？", "tr": "5 volt kaç milivolt eder?", "sw": "Volt 5 ni millivolt ngapi?", "ka": "5 ვოლტი რამდენ მილივოლტს უდრის?"},
    9: {"en": "How many hertz are in 2 megahertz?", "el": "Πόσα Hertz είναι 2 MHz;", "zh": "2 MHz 等于多少赫兹？", "tr": "2 megahertz kaç hertz eder?", "sw": "Megahertz 2 ni hertz ngapi?", "ka": "2 მეგაჰერცი რამდენ ჰერცს უდრის?"},
    10: {"en": "How many millinewtons are in 2 newtons?", "el": "Πόσα millinewton είναι 2 newton;", "zh": "2 牛顿是多少毫牛顿？", "tr": "2 newton kaç milinewton eder?", "sw": "Newton 2 ni millinewton ngapi?", "ka": "2 ნიუტონი რამდენ მილინიუტონს უდრის?"},
    11: {"en": "How many watts are in 2 kilowatts?", "el": "Πόσα Watt είναι 2 kW;", "zh": "2 千瓦是多少瓦？", "tr": "2 kilowatt kaç watt eder?", "sw": "Kilowatt 2 ni watt ngapi?", "ka": "2 კილოვატი რამდენ ვატს უდრის?"},
    12: {"en": "How many millitesla are in 3 Teslas?", "el": "Πόσα millitesla είναι 3 Tesla;", "zh": "3 特斯拉是多少毫特斯拉？", "tr": "3 tesla kaç militesla eder?", "sw": "Tesla 3 ni millitesla ngapi?", "ka": "3 ტესლა რამდენ მილიტესლას უდრის?"},
    13: {"en": "What is the area of 2 hectares in square meters?", "el": "Ποια είναι η επιφάνεια 2 εκταρίων σε τετραγωνικά μέτρα;", "zh": "2 公顷等于多少平方米？", "tr": "2 hektar kaç metrekaredir?", "sw": "Hekta 2 ni mita mraba ngapi?", "ka": "2 ჰექტარი რამდენ კვადრატულ მეტრს უდრის?"},
    14: {"en": "How many lux are equivalent to 4 lumens per square meter?", "el": "Πόσα lux αντιστοιχούν σε 4 lumen/m²;", "zh": "4 流明/平方米等于多少勒克斯？", "tr": "4 lümen/metrekare kaç lux eder?", "sw": "Lumen 4 kwa mita mraba ni lux ngapi?", "ka": "4 ლუმენი/მ² რამდენ ლუქსს უდრის?"},
    15: {"en": "How many kilometers are in 2 light years?", "el": "Πόσα χιλιόμετρα είναι 2 έτη φωτός;", "zh": "2 光年是多少公里？", "tr": "2 ışık yılı kaç kilometredir?", "sw": "Miaka ya mwanga 2 ni kilomita ngapi?", "ka": "2 სინათლის წელი რამდენ კილომეტრს უდრის?"},
    16: {"en": "How many bits are in 3 bytes?", "el": "Πόσα bits είναι 3 bytes;", "zh": "3 字节是多少比特？", "tr": "3 byte kaç bit eder?", "sw": "Byte 3 ni bit ngapi?", "ka": "3 ბაიტი რამდენ ბიტს უდრის?"}
}

QUESTION_CONFIG["level2"] = {
    1: {"en": "A stopwatch runs for 3 and a half minutes. How many seconds does it count?", "el": "Ένα χρονόμετρο τρέχει για 3,5 λεπτά. Πόσα δευτερόλεπτα μετράει;", "zh": "秒表运行了 3 分半钟。它计了多少秒？", "tr": "Bir kronometre 3 buçuk dakika çalışıyor. Kaç saniye sayar?", "sw": "Saa ya kusimama inafanya kazi kwa dakika 3 na nusu. Inahisabu sekunde ngapi?", "ka": "წამზომი მუშაობს 3 წუთნახევარი. რამდენ წამს ითვლის?"},
    2: {"en": "A person weighs 72 kilograms. What is the persons weight in grams?", "el": "Ένα άτομο ζυγίζει 72 κιλά. Ποιο είναι το βάρος του σε γραμμάρια;", "zh": "一个人体重 72 公斤。这个人的体重是多少克？", "tr": "Bir kişi 72 kilogram ağırlığındadır. Kişinin ağırlığı gram cinsinden nedir?", "sw": "Mtu ana uzito wa kilo 72. Uzito wa mtu huyo ni gramu ngapi?", "ka": "ადამიანი იწონის 72 კილოგრამს. რა არის ადამიანის წონა გრამებში?"},
    3: {"en": "A circular track has a circumference of 400 meters. What is its diameter in centimeters?", "el": "Μια κυκλική διαδρομή έχει περιφέρεια 400 μέτρα. Ποια είναι η διάμετρός της σε εκατοστά;", "zh": "一个圆形跑道的周长是 400 米。它的直径是多少厘米？", "tr": "Dairesel bir pistin çevresi 400 metredir. Santimetre cinsinden çapı nedir?", "sw": "Njia ya duara ina mzingo wa mita 400. Kipenyo chake ni sentimita ngapi?", "ka": "წრიული ტრასის გარშემოწერილობაა 400 მეტრი. რა არის მისი დიამეტრი სანტიმეტრებში?"},
    4: {"en": "Water boils at 100°C. What is its boiling point in Kelvin?", "el": "Το νερό βράζει στους 100°C. Ποιο είναι το σημείο βρασμού του σε Kelvin;", "zh": "水在 100°C 沸腾。它的沸点在开尔文是多少？", "tr": "Su 100°C'de kaynar. Kelvin cinsinden kaynama noktası nedir?", "sw": "Maji yanachemka kwa 100°C. Kiwango chake cha kuchemka ni Kelvin gani?", "ka": "წყალი დუღს 100°C-ზე. რა არის მისი დუღილის წერტილი კელვინში?"},
    5: {"en": "If you have a container that holds 1,250 milliliters of liquid, how many cubic centimeters of liquid can it hold?", "el": "Εάν έχετε ένα δοχείο που χωράει 1.250 ml υγρού, πόσα κυβικά εκατοστά υγρού μπορεί να χωρέσει;", "zh": "如果你有一个盛放 1,250 毫升液体的容器，它可以盛放多少立方厘米的液体？", "tr": "1.250 mililitre sıvı alan bir kabınız varsa, bu kap kaç santimetreküp sıvı alabilir?", "sw": "Ikiwa una chombo kinachoshika mililita 1,250 za kioevu, kinaweza kushika sentimita za ujazo ngapi za kioevu?", "ka": "თუ გაქვთ კონტეინერი, რომელიც იტევს 1250 მილილიტრ სითხეს, რამდენ კუბურ სანტიმეტრ სითხეს იტევს ის?"},
    6: {"en": "A person burns 200 Joules of energy while jogging. How many calories did they burn?", "el": "Ένα άτομο καίει 200 Joule ενέργειας τρέχοντας. Πόσες θερμίδες έκαψε;", "zh": "一个人在慢跑时消耗了 200 焦耳的能量。他们消耗了多少卡路里？", "tr": "Bir kişi koşu yaparken 200 Joule enerji yakıyor. Kaç kalori yaktı?", "sw": "Mtu anachoma Joule 200 za nishati wakati wa kukimbia. Walichoma kalori ngapi?", "ka": "ადამიანი სირბილისას წვავს 200 ჯოულ ენერგიას. რამდენი კალორია დაწვა მან?"},
    7: {"en": "A diver is 100 meters below the surface of the ocean where the pressure is 152,300 Pascals. How many atmospheres of pressure are they experiencing?", "el": "Ένας δύτης βρίσκεται 100 μέτρα κάτω από την επιφάνεια όπου η πίεση είναι 152.300 Pa. Πόσες ατμόσφαιρες πίεσης δέχεται;", "zh": "一名潜水员位于海平面以下 100 米处，那里的压力为 152,300 帕斯卡。他们承受着多少个大气压？", "tr": "Bir dalgıç, basıncın 152.300 Pascal olduğu okyanus yüzeyinin 100 metre altındadır. Kaç atmosfer basınç yaşıyorlar?", "sw": "Mzamiaji yuko mita 100 chini ya uso wa bahari ambapo shinikizo ni Pascal 152,300. Je, wanapata shinikizo la anga ngapi?", "ka": "მყვინთავი იმყოფება ოკეანის ზედაპირიდან 100 მეტრის სიღრმეზე, სადაც წნევა 152,300 პასკალია. რამდენი ატმოსფეროს წნევას განიცდის იგი?"},
    8: {"en": "A circuit is powered by 30,000 millivolts. How many volts is this?", "el": "Ένα κύκλωμα τροφοδοτείται με 30.000 millivolt. Πόσα volt είναι αυτό;", "zh": "一个电路的供电电压为 30,000 毫伏。这是多少伏特？", "tr": "Bir devre 30.000 milivolt ile besleniyor. Bu kaç volttur?", "sw": "Mzunguko unaendeshwa na millivolt 30,000. Hii ni volt ngapi?", "ka": "წრედი იკვებება 30,000 მილივოლტით. რამდენი ვოლტია ეს?"},
    9: {"en": "An oscillator operates at 4 megahertz. What is the period of the wave in seconds?", "el": "Ένας ταλαντωτής λειτουργεί στα 4 MHz. Ποια είναι η περίοδος του κύματος σε δευτερόλεπτα;", "zh": "一个振荡器的工作频率为 4 兆赫。波的周期是多少秒？", "tr": "Bir osilatör 4 megahertz'de çalışıyor. Dalganın saniye cinsinden periyodu nedir?", "sw": "Oscillator inafanya kazi kwa megahertz 4. Kipindi cha wimbi katika sekunde ni gani?", "ka": "ოსცილატორი მუშაობს 4 მეგაჰერცზე. რა არის ტალღის პერიოდი წამებში?"},
    10: {"en": "A person applies a force of 24 newtons to a cart with a mass of 3 kilograms. What is the is the force applied to the cart by the person in millinewtons?", "el": "Ένα άτομο ασκεί δύναμη 24 Newton σε ένα καρότσι μάζας 3 kg. Ποια είναι η δύναμη σε millinewton;", "zh": "一个人对质量为 3 公斤的小车施加 24 牛顿的力。此人对小车施加的力是多少毫牛顿？", "tr": "Bir kişi 3 kg kütleli bir arabaya 24 newton kuvvet uyguluyor. Uygulanan kuvvet milinewton cinsinden nedir?", "sw": "Mtu anatumia nguvu ya newton 24 kwenye toroli lenye uzito wa kilo 3. Nguvu inayotumiwa ni millinewton ngapi?", "ka": "ადამიანი იყენებს 24 ნიუტონის ძალას 3 კილოგრამიანი მასის ურიკაზე. რა არის გამოყენებული ძალა მილინიუტონებში?"},
    11: {"en": "A lightbulb consumes 900 watts of power. How many kilowatts is this?", "el": "Ένας λαμπτήρας καταναλώνει 900 Watt. Πόσα kW είναι αυτό;", "zh": "一个灯泡消耗 900 瓦的功率。这是多少千瓦？", "tr": "Bir ampul 900 watt güç tüketiyor. Bu kaç kilowatt eder?", "sw": "Balbu ya mwanga inatumia wati 900 za nguvu. Hii ni kilowati ngapi?", "ka": "ნათურა მოიხმარს 900 ვატ სიმძლავრეს. რამდენი კილოვატია ეს?"},
    12: {"en": "A coil generates a magnetic field of 300 millitesla. What is this field strength in Teslas?", "el": "Ένα πηνίο παράγει μαγνητικό πεδίο 300 millitesla. Ποια είναι η ισχύς του σε Tesla;", "zh": "线圈产生 300 毫特斯拉的磁场。特斯拉中的磁场强度是多少？", "tr": "Bir bobin 300 militesla manyetik alan üretir. Tesla cinsinden bu alan gücü nedir?", "sw": "Koili inazalisha uga wa sumaku wa millitesla 300. Nguvu ya uga huu katika Tesla ni gani?", "ka": "კოჭა წარმოქმნის 300 მილიტესლა მაგნიტურ ველს. რა არის ამ ველის სიძლიერე ტესლაში?"},
    13: {"en": "A park has an area of 86,000 square meters. How many hectares is the park?", "el": "Ένα πάρκο έχει έκταση 86.000 τ.μ. Πόσα εκτάρια είναι το πάρκο;", "zh": "一个公园的面积为 86,000 平方米。这个公园有多少公顷？", "tr": "Bir parkın alanı 86.000 metrekaredir. Park kaç hektardır?", "sw": "Hifadhi ina eneo la mita za mraba 86,000. Hifadhi hiyo ina hekta ngapi?", "ka": "პარკის ფართობია 86,000 კვადრატული მეტრი. რამდენი ჰექტარია პარკი?"},
    14: {"en": "A workspace is illuminated at a level of 6 lux. What is the illumination in lumens per square meter?", "el": "Ένας χώρος εργασίας φωτίζεται με 6 lux. Ποιος είναι ο φωτισμός σε lumen/m²;", "zh": "一个工作区的照度为 6 勒克斯。每平方米多少流明？", "tr": "Bir çalışma alanı 6 lux seviyesinde aydınlatılmaktadır. Metrekare başına lümen cinsinden aydınlatma nedir?", "sw": "Eneo la kazi limemulika kwa kiwango cha lux 6. Mwangaza katika lumen kwa kila mita ya mraba ni gani?", "ka": "სამუშაო სივრცე განათებულია 6 ლუქსის დონეზე. რა არის განათება ლუმენებში კვადრატულ მეტრზე?"},
    15: {"en": "The Andromeda Galaxy is approximately 23 light years from Earth. What is this distance in kilometers?", "el": "Ο Γαλαξίας της Ανδρομέδας απέχει περίπου 23 έτη φωτός. Ποια είναι αυτή η απόσταση σε χιλιόμετρα;", "zh": "仙女座星系距离地球约 23 光年。这个距离是多少公里？", "tr": "Andromeda Galaksisi Dünya'dan yaklaşık 23 ışık yılı uzaklıktadır. Bu mesafe kaç kilometredir?", "sw": "Galaksi ya Andromeda iko umbali wa miaka ya mwanga 23 kutoka Duniani. Umbali huu ni kilomita ngapi?", "ka": "ანდრომედას გალაქტიკა დედამიწიდან დაახლოებით 23 სინათლის წლით არის დაშორებული. რა არის ეს მანძილი კილომეტრებში?"},
    16: {"en": "If a document is 8,000 bits in size, how many bytes does it occupy?", "el": "Αν ένα έγγραφο έχει μέγεθος 8.000 bits, πόσα bytes καταλαμβάνει;", "zh": "如果一个文档的大小为 8,000 比特，它占用多少字节？", "tr": "Bir belge 8.000 bit boyutundaysa, kaç byte yer kaplar?", "sw": "Ikiwa hati ina ukubwa wa biti 8,000, inachukua baiti ngapi?", "ka": "თუ დოკუმენტის ზომაა 8,000 ბიტი, რამდენ ბაიტს იკავებს ის?"}
}

QUESTION_CONFIG["level3"] = {
    1: {"en": "A marathon runner runs at a speed of 170 meters per minute. How many seconds will it take them to complete a 42-kilometer race?", "el": "Ένας μαραθωνοδρόμος τρέχει με 170 m/min. Πόσα δευτερόλεπτα θα χρειαστεί για 42 km;", "zh": "一名马拉松跑者的跑步速度为每分钟 170 米。完成 42 公里的比赛需要多少秒？", "tr": "Bir maraton koşucusu dakikada 170 metre hızla koşuyor. 42 kilometrelik bir yarışı tamamlaması kaç saniye sürer?", "sw": "Mkimbiaji wa marathon anakimbia kwa kasi ya mita 170 kwa dakika. Itamchukua sekunde ngapi kumaliza mbio za kilomita 42?", "ka": "მარათონელი დარბის 170 მეტრი/წუთში სიჩქარით. რამდენი წამი დასჭირდება მას 42 კილომეტრიანი რბოლის დასასრულებლად?"},
    2: {"en": "A vehicle's engine weighs 650 kilograms. If 15% of the weight is aluminum, what is the weight of the aluminum in grams?", "el": "Κινητήρας οχήματος ζυγίζει 650 kg. Αν το 15% είναι αλουμίνιο, ποιο είναι το βάρος του σε γραμμάρια;", "zh": "一辆汽车的发动机重 650 公斤。如果重量的 15% 是铝，那么铝的重量是多少克？", "tr": "Bir aracın motoru 650 kg ağırlığındadır. %15'i alüminyum ise, alüminyumun ağırlığı gram cinsinden nedir?", "sw": "Injini ya gari ina uzito wa kilo 650. Ikiwa 15% ya uzito ni alumini, uzito wa alumini ni gramu ngapi?", "ka": "ავტომობილის ძრავა იწონის 650 კილოგრამს. თუ წონის 15% ალუმინია, რა არის ალუმინის წონა გრამებში?"},
    3: {"en": "If a rectangular field is 50 meters long and 30 meters wide, what is its area in square centimeters?", "el": "Αν ένα ορθογώνιο χωράφι έχει μήκος 50 m και πλάτος 30 m, ποιο είναι το εμβαδόν του σε τ.εκ.;", "zh": "如果一个长方形田地长 50 米，宽 30 米，它的面积是多少平方厘米？", "tr": "Dikdörtgen bir alan 50 metre uzunluğunda ve 30 metre genişliğinde ise, santimetrekare cinsinden alanı nedir?", "sw": "Ikiwa uwanja wa mstatili una urefu wa mita 50 na upana wa mita 30, eneo lake ni sentimita za mraba ngapi?", "ka": "თუ მართკუთხა მინდორი არის 50 მეტრი სიგრძის და 30 მეტრი სიგანის, რა არის მისი ფართობი კვადრატულ სანტიმეტრებში?"},
    4: {"en": "At a certain point in time, the temperature of a black hole's event horizon is measured to be 20°C. If the temperature in Celsius decreases by 30% after an event, what is the new temperature in Kelvin?", "el": "Η θερμοκρασία ενός ορίζοντα γεγονότων είναι 20°C. Αν μειωθεί κατά 30% μετά από ένα συμβάν, ποια είναι η νέα θερμοκρασία σε Kelvin;", "zh": "在某个时间点，测量到黑洞事件视界的温度为 20°C。如果事件发生后摄氏温度下降 30%，新的开尔文温度是多少？", "tr": "Bir kara deliğin olay ufkunun sıcaklığı 20°C'dir. Bir olaydan sonra sıcaklık %30 azalırsa, Kelvin cinsinden yeni sıcaklık nedir?", "sw": "Katika wakati fulani, joto la upeo wa tukio la shimo nyeusi linapimwa kuwa 20°C. Ikiwa joto katika Selsiasi linapungua kwa 30% baada ya tukio, joto jipya katika Kelvin ni gani?", "ka": "დროის გარკვეულ მომენტში, შავი ხვრელის მოვლენათა ჰორიზონტის ტემპერატურა იზომება 20°C-ად. თუ მოვლენის შემდეგ ცელსიუსის ტემპერატურა 30%-ით შემცირდება, რა არის ახალი ტემპერატურა კელვინში?"},
    5: {"en": "A spherical ball has a radius of 10 cm. What is its volume in milliliters?", "el": "Μια σφαιρική μπάλα έχει ακτίνα 10 cm. Ποιος είναι ο όγκος της σε ml;", "zh": "一个半径为 10 厘米的球体。它的体积是多少毫升？", "tr": "Yarıçapı 10 cm olan küresel bir topun hacmi kaç mililitredir?", "sw": "Mpira wa duara una rasi ya sm 10. Kiasi chake katika mililita ni gani?", "ka": "სფერული ბურთის რადიუსია 10 სმ. რა არის მისი მოცულობა მილილიტრებში?"},
    6: {"en": "A car burns 3,400 Joules of fuel every minute. If the car runs for 2 hours, how many calories does it burn?", "el": "Ένα αυτοκίνητο καίει 3.400 J καυσίμου ανά λεπτό. Αν τρέχει για 2 ώρες, πόσες θερμίδες καίει;", "zh": "一辆汽车每分钟燃烧 3,400 焦耳燃料。如果汽车行驶 2 小时，它消耗多少卡路里？", "tr": "Bir araba her dakika 3.400 Joule yakıt yakıyor. Araba 2 saat çalışırsa kaç kalori yakar?", "sw": "Gari linachoma Joule 3,400 za mafuta kila dakika. Ikiwa gari litafanya kazi kwa saa 2, linachoma kalori ngapi?", "ka": "ავტომობილი წვავს 3,400 ჯოულ საწვავს ყოველ წუთში. თუ ავტომობილი მუშაობს 2 საათის განმავლობაში, რამდენ კალორიას წვავს ის?"},
    7: {"en": "A pressurized gas tank holds a gas at a pressure of 150,000 Pascals. If the gas occupies a volume of 4 cubic meters at this pressure, and the gas is suddenly released to 2 atmospheres, what will be the new volume of the gas? Assume temperature and the number of gas molecules remain constant and use Boyle's Law.", "el": "Δεξαμενή αερίου έχει πίεση 150.000 Pa και όγκο 4 m³. Αν η πίεση γίνει 2 atm, ποιος θα είναι ο νέος όγκος; (Νόμος Boyle)", "zh": "一个压力储气罐中的气体压力为 150,000 帕斯卡。如果气体在此压力下占据 4 立方米的体积，并突然释放到 2 个大气压，气体的体积将变为多少？假设温度和气体分子数保持不变，使用波义耳定律。", "tr": "Basınçlı bir gaz tankı 150.000 Pascal basınçta gaz tutmaktadır. Gaz bu basınçta 4 metreküp hacim kaplıyorsa ve aniden 2 atmosfere bırakılırsa, gazın yeni hacmi ne olur? Sıcaklığın ve gaz moleküllerinin sayısının sabit kaldığını varsayın ve Boyle Yasasını kullanın.", "sw": "Tanki la gesi lililoshinikizwa linashikilia gesi kwa shinikizo la Pascal 150,000. Ikiwa gesi inachukua kiasi cha mita za ujazo 4 kwa shinikizo hili, na gesi inatolewa ghafla kwa anga 2, kiasi kipya cha gesi kitakuwa nini? Chukulia joto na idadi ya molekuli za gesi zinabaki thabiti na utumie Sheria ya Boyle.", "ka": "წნევის ქვეშ მყოფი გაზის ავზი შეიცავს გაზს 150,000 პასკალი წნევით. თუ გაზი იკავებს 4 კუბურ მეტრ მოცულობას ამ წნევაზე და გაზი მოულოდნელად განთავისუფლდება 2 ატმოსფერომდე, რა იქნება გაზის ახალი მოცულობა? ჩათვალეთ, რომ ტემპერატურა და გაზის მოლეკულების რაოდენობა მუდმივი რჩება და გამოიყენეთ ბოილის კანონი."},
    8: {"en": "A battery supplies 100,000 millivolts to a device. If the device operates with a resistance of 20 ohms, what is the current (in Amperes) flowing through the device using Ohm's Law?", "el": "Μια μπαταρία παρέχει 100.000 millivolt σε μια συσκευή με αντίσταση 20 Ohm. Ποια είναι η ένταση του ρεύματος (Ampere) κατά τον νόμο του Ohm;", "zh": "一个电池向一台设备供应 100,000 毫伏。如果该设备的工作电阻为 20 欧姆，根据欧姆定律，流过该设备的电流（以安培为单位）是多少？", "tr": "Bir pil bir cihaza 100.000 milivolt besliyor. Cihaz 20 ohm dirençle çalışıyorsa, Ohm Yasasını kullanarak cihazdan geçen akım (Amper cinsinden) nedir?", "sw": "Betri inatoa millivolt 100,000 kwa kifaa. Ikiwa kifaa kinafanya kazi na upinzani wa ohm 20, mkondo wa umeme (katika Ampere) unaopita kwenye kifaa ni nini kwa kutumia Sheria ya Ohm?", "ka": "ბატარეა აწვდის 100,000 მილივოლტს მოწყობილობას. თუ მოწყობილობა მუშაობს 20 ომიანი წინააღმდეგობით, რა არის მოწყობილობაში გამავალი დენი (ამპერებში) ომის კანონის გამოყენებით?"},
    9: {"en": "A circuit has a signal with a frequency of 6 megahertz. What is the wavelength of the signal if the speed of light is approximately 3×10^8 meters per second?", "el": "Ένα κύκλωμα έχει σήμα 6 MHz. Ποιο είναι το μήκος κύματος αν η ταχύτητα του φωτός είναι 3×10⁸ m/s;", "zh": "一个电路信号频率为 6 兆赫。如果光速约为 3×10^8 米/秒，信号波长是多少？", "tr": "Bir devrede 6 megahertz frekansında bir sinyal vardır. Işık hızı yaklaşık 3×10^8 metre/saniye ise sinyalin dalga boyu nedir?", "sw": "Mzunguko una ishara yenye masafa ya megahertz 6. Urefu wa wimbi la ishara ni nini ikiwa kasi ya mwanga ni takriban mita 3×10^8 kwa sekunde?", "ka": "წრედს აქვს სიგნალი 6 მეგაჰერცი სიხშირით. რა არის სიგნალის ტალღის სიგრძე, თუ სინათლის სიჩქარე დაახლოებით 3×10^8 მეტრი წამშია?"},
    10: {"en": "A 10-kilogram object is pulled with a force of 4,300 millinewtons. What is the acceleration of the object (in meters per second squared)?", "el": "Αντικείμενο 10 kg έλκεται με δύναμη 4.300 millinewton. Ποια είναι η επιτάχυνσή του (m/s²);", "zh": "一个 10 公斤重的物体受到 4,300 毫牛顿的力。物体的加速度（单位为米/秒平方）是多少？", "tr": "10 kilogramlık bir nesne 4.300 milinewtonluk bir kuvvetle çekiliyor. Nesnenin ivmesi (metre/saniye kare cinsinden) nedir?", "sw": "Kitu chenye uzito wa kilo 10 kinavutwa kwa nguvu ya millinewton 4,300. Kasi ya kuongezeka kwa kitu hicho (katika mita kwa kila sekunde ya mraba) ni nini?", "ka": "10-კილოგრამიანი ობიექტი იწევა 4,300 მილინიუტონი ძალით. რა არის ობიექტის აჩქარება (მეტრებში წამში კვადრატში)?"},
    11: {"en": "A factory uses 12 kilowatts for 10 hours per day for 30 days. What is the total energy consumption in watt-hours?", "el": "Ένα εργοστάσιο χρησιμοποιεί 12 kW για 10 ώρες/μέρα επί 30 μέρες. Ποια είναι η συνολική κατανάλωση σε Watt-ώρες;", "zh": "一家工厂每天使用 12 千瓦的功率，每天运行 10 小时，持续 30 天。总能耗是多少瓦时？", "tr": "Bir fabrika 30 gün boyunca günde 10 saat 12 kilowatt kullanıyor. Watt-saat cinsinden toplam enerji tüketimi nedir?", "sw": "Kiwanda kinatumia kilowati 12 kwa saa 10 kwa siku kwa siku 30. Jumla ya matumizi ya nishati katika wati-saa ni gani?", "ka": "ქარხანა მოიხმარს 12 კილოვატს დღეში 10 საათის განმავლობაში 30 დღის განმავლობაში. რა არის ჯამური ენერგიის მოხმარება ვატ-საათებში?"},
    12: {"en": "A particle moves through a magnetic field of 3,600 millitesla with a charge of 2×10^(−6) C and a velocity of 10^5 m/s. What is the magnetic force on the particle?", "el": "Σωματίδιο κινείται σε πεδίο 3.600 mT με φορτίο 2×10⁻⁶ C και ταχύτητα 10⁵ m/s. Ποια είναι η μαγνητική δύναμη;", "zh": "一个电荷量为 2×10^(−6) C、速度为 10^5 m/s 的粒子穿过 3,600 毫特斯拉的磁场。粒子受到的磁力是多少？", "tr": "Bir parçacık, 2×10^(−6) C yüke ve 10^5 m/s hıza sahip olarak 3.600 militesla manyetik alandan geçiyor. Parçacık üzerindeki manyetik kuvvet nedir?", "sw": "Chembe inapita kwenye uga wa sumaku wa millitesla 3,600 ikiwa na chaji ya 2×10^(−6) C na kasi ya 10^5 m/s. Nguvu ya sumaku kwenye chembe hiyo ni gani?", "ka": "ნაწილაკი მოძრაობს 3600 მილიტესლა მაგნიტურ ველში 2×10^(−6) C მუხტით და 10^5 მ/წმ სიჩქარით. რა არის მაგნიტური ძალა ნაწილაკზე?"},
    13: {"en": "A triangular plot of land has a base of 300 meters and a height of 350 meters. How many hectares is the plot?", "el": "Τριγωνικό οικόπεδο έχει βάση 300 m και ύψος 350 m. Πόσα εκτάρια είναι το οικόπεδο;", "zh": "一块三角形土地的底边为 300 米，高为 350 米。这块地有多少公顷？", "tr": "Üçgen bir arazi parçasının tabanı 300 metre ve yüksekliği 350 metredir. Arazi kaç hektardır?", "sw": "Kiwanja cha pembetatu kina kitako cha mita 300 na kimo cha mita 350. Kiwanja hicho kina hekta ngapi?", "ka": "სამკუთხა მიწის ნაკვეთის ფუძეა 300 მეტრი, ხოლო სიმაღლე - 350 მეტრი. რამდენი ჰექტარია ნაკვეთი?"},
    14: {"en": "A light source emits 300 lumens uniformly over a circular area with a radius of 10 meters. What is the average illumination in lux over this area?", "el": "Πηγή εκπέμπει 300 lumen ομοιόμορφα σε κύκλο ακτίνας 10 m. Ποιος είναι ο μέσος φωτισμός σε lux;", "zh": "一个光源在半径为 10 米的圆形区域内均匀发射 300 流明。该区域的平均照度是多少勒克斯？", "tr": "Bir ışık kaynağı, 10 metre yarıçaplı dairesel bir alana düzgün bir şekilde 300 lümen yaymaktadır. Ortalama aydınlatma lux cinsinden nedir?", "sw": "Chanzo cha mwanga hutoa lumen 300 kwa usawa kwenye eneo la duara lenye rasi ya mita 10. Mwangaza wa wastani katika lux kwenye eneo hili ni gani?", "ka": "სინათლის წყარო ასხივებს 300 ლუმენს თანაბრად წრიულ ფართობზე, რომლის რადიუსია 10 მეტრი. რა არის საშუალო განათება ლუქსში ამ ფართობზე?"},
    15: {"en": "A black hole is 150 light years away. If light travels at a speed of 0.3 billion kilometers per second, how long would it take for light to travel this distance in seconds?", "el": "Μια μαύρη τρύπα απέχει 150 έτη φωτός. Αν το φως τρέχει με 0,3 δις km/s, πόσα δευτερόλεπτα θα χρειαστεί;", "zh": "一个黑洞距离 150 光年。如果光以每秒 3 亿公里的速度传播，光走完这段距离需要多少秒？", "tr": "Bir kara delik 150 ışık yılı uzaklıktadır. Işık saniyede 0,3 milyar kilometre hızla hareket ederse, bu mesafeyi kat etmesi kaç saniye sürer?", "sw": "Shimo nyeusi liko umbali wa miaka ya mwanga 150. Ikiwa mwanga unasafiri kwa kasi ya kilomita bilioni 0.3 kwa sekunde, itachukua sekunde ngapi?", "ka": "შავი ხვრელი 150 სინათლის წლით არის დაშორებული. თუ სინათლე მოძრაობს 0.3 მილიარდი კილომეტრი წამში სიჩქარით, რამდენი წამი დასჭირდება სინათლეს ამ მანძილის გასავლელად?"},
    16: {"en": "A 1-minute high-definition video uses a data rate of 8×10^6 bytes per second. How many bits does the video consume in total?", "el": "Βίντεο HD διάρκειας 1 λεπτού έχει ρυθμό 8×10⁶ bytes/s. Πόσα bits καταναλώνει συνολικά;", "zh": "一段 1 分钟的高清视频使用每秒 8×10^6 字节的数据速率。视频总共消耗多少比特？", "tr": "1 dakikalık yüksek çözünürlüklü bir video saniyede 8×10^6 byte veri hızı kullanır. Video toplamda kaç bit tüketir?", "sw": "Video ya dakika 1 ya ubora wa juu inatumia kasi ya data ya baiti 8×10^6 kwa sekunde. Video hiyo inatumia jumla ya biti ngapi?", "ka": "1-წუთიანი მაღალი გარჩევადობის ვიდეო იყენებს მონაცემთა სიჩქარეს 8×10^6 ბაიტი წამში. რამდენი ბიტს მოიხმარს ვიდეო სულ?"}
}

def build_prompt(worldA_text, worldB_text, question, target_world, lang):
    cfg = LANG_CONFIG[lang]
    return f"""{worldA_text}\n\n{worldB_text}\n\n{cfg["answer"]}\n\nQuestion: {question} in {target_world}\n\nInstructions:\n{cfg["instruction"]}\n{cfg["return_line"]}\n\n{cfg["format"]}\n{cfg["final"]}"""

df_level1 = pd.read_excel('units.xlsx', sheet_name='level1', engine='openpyxl')
df_level2 = pd.read_excel('units.xlsx', sheet_name='level2', engine='openpyxl')
df_level3 = pd.read_excel('units.xlsx', sheet_name='level3', engine='openpyxl')

model_id = "/project/home/p201200/MultiWorld/qwen_model/72B"
print(f"Loading model: {model_id}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

output_dir = "/project/home/p201200/MultiWorld/outputs"
os.makedirs(output_dir, exist_ok=True)

levels = {
    1: df_level1,
    2: df_level2,
    3: df_level3
}

variants = {
    "a": {"valueA_col": "Y1", "valueB_col": "real_value"},
    "b": {"valueA_col": "real_value", "valueB_col": "Y3"}
}
for level_num, df in levels.items():
    if level_num ==1:
        max_tokens=50
    elif level_num ==2:
        max_tokens=60
    else:
        max_tokens=100
    print(f"\n{'='*30}\n Starting Level {level_num}\n{'='*30}")

    for variant_name, config in variants.items():
        print(f"\n--- Running Variant {variant_name} ---")
        current_results = [] 

        for index, row in df.iterrows():
            constant, valueA, valueB, question_en = row['X'], row[config["valueA_col"]], row[config["valueB_col"]], row['question']

            h_lang = random.choice(LANG_GROUPS["high"])
            m_lang = random.choice(LANG_GROUPS["medium"])
            l_lang = random.choice(LANG_GROUPS["low"])

            selected_pairs = [
                (h_lang, m_lang, "High-Medium"),
                (m_lang, l_lang, "Medium-Low"),
                (h_lang, l_lang, "High-Low")
            ]

            for lang1, lang2, pair_cat in selected_pairs:
                wA_text_L1 = f"WorldA: {WORLD_CONFIG[lang1].format(constant=constant, value=valueA)}"
                wB_text_L2 = f"WorldB: {WORLD_CONFIG[lang2].format(constant=constant, value=valueB)}"
                for current_lang in [lang1, lang2]:
                    q_task = QUESTION_CONFIG[f"level{level_num}"][index + 1][current_lang]

                    for target_world in ["WorldA", "WorldB"]:
                        prompt = build_prompt(wA_text_L1, wB_text_L2, q_task, target_world, current_lang)

                        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                        with torch.no_grad():
                            output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

                        raw_output = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

                        res_entry = {
                            "ID": index + 1,
                            "Const": constant,
                            "Var": variant_name,
                            "Pair": pair_cat,
                            "Lang": current_lang,
                            "Target": target_world,
                            "Raw": raw_output,
                            "EN": translate_to_english(raw_output)
                        }
                        current_results.append(res_entry)

        output_csv = os.path.join(output_dir, f"qwen72_lang_units_zs_{level_num}{variant_name}.csv")
        pd.DataFrame(current_results).to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

print("All runs completed.")
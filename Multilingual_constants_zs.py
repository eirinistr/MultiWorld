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
        1: {
            "en": "What is the first digit of pi?",
            "el": "Ποιο είναι το πρώτο ψηφίο του π;",
            "zh": "π 的第一位数字是多少？",
            "tr": "Pi sayısının ilk basamağı nedir?",
            "sw": "Ni tarakimu gani ya kwanza ya pi?",
            "ka": "რა არის π-ის პირველი ციფრი?"
        },
        2: {
            "en": "What is the first digit of e?",
            "el": "Ποιο είναι το πρώτο ψηφίο του e;",
            "zh": "e 的第一位数字是多少？",
            "tr": "e sayısının ilk basamağı nedir?",
            "sw": "Ni tarakimu gani ya kwanza ya e?",
            "ka": "რა არის e-ის პირველი ციფრი?"
        },
        3: {
            "en": "What is the first digit of phi?",
            "el": "Ποιο είναι το πρώτο ψηφίο του φ;",
            "zh": "φ 的第一位数字是多少？",
            "tr": "phi sayısının ilk basamağı nedir?",
            "sw": "Ni tarakimu gani ya kwanza ya phi?",
            "ka": "რა არის φ-ის პირველი ციფრი?"
        },
        4: {
            "en": "How far does light travel in one second?",
            "el": "Πόσο μακριά ταξιδεύει το φως σε ένα δευτερόλεπτο;",
            "zh": "光在一秒钟内传播多远？",
            "tr": "Işık bir saniyede ne kadar yol alır?",
            "sw": "Mwanga husafiri umbali gani kwa sekunde moja?",
            "ka": "რამდენ მანძილს გადის სინათლე ერთ წამში?"
        },
        5: {
            "en": "What is the first non-zero digit of the gravitational constant?",
            "el": "Ποιο είναι το πρώτο μη μηδενικό ψηφίο της βαρυτικής σταθεράς;",
            "zh": "引力常数的第一个非零数字是多少？",
            "tr": "Yerçekimi sabitinin ilk sıfır olmayan basamağı nedir?",
            "sw": "Ni tarakimu gani ya kwanza isiyo sifuri ya konstanti ya mvuto?",
            "ka": "რა არის გრავიტაციის მუდმივის პირველი არანულოვანი ციფრი?"
        },
        6: {
            "en": "What is the first non-zero digit of Planck's constant?",
            "el": "Ποιο είναι το πρώτο μη μηδενικό ψηφίο της σταθεράς του Planck;",
            "zh": "普朗克常数的第一个非零数字是多少？",
            "tr": "Planck sabitinin ilk sıfır olmayan basamağı nedir?",
            "sw": "Ni tarakimu gani ya kwanza isiyo sifuri ya konstanti ya Planck?",
            "ka": "რა არის პლანკის მუდმივის პირველი არანულოვანი ციფრი?"
        },
        7: {
            "en": "What is the first non-zero digit of the elementary charge?",
            "el": "Ποιο είναι το πρώτο μη μηδενικό ψηφίο του στοιχειώδους φορτίου;",
            "zh": "基本电荷的第一个非零数字是多少？",
            "tr": "Temel yükün ilk sıfır olmayan basamağı nedir?",
            "sw": "Ni tarakimu gani ya kwanza isiyo sifuri ya chaji ya msingi?",
            "ka": "რა არის ელემენტარული მუხტის პირველი არანულოვანი ციფრი?"
        },
        8: {
            "en": "What is the first digit of Avogadro's number?",
            "el": "Ποιο είναι το πρώτο ψηφίο του αριθμού Avogadro;",
            "zh": "阿伏伽德罗常数的第一位数字是多少？",
            "tr": "Avogadro sayısının ilk basamağı nedir?",
            "sw": "Ni tarakimu gani ya kwanza ya nambari ya Avogadro?",
            "ka": "რა არის ავოგადროს რიცხვის პირველი ციფრი?"
        },
        9: {
            "en": "What is the first non-zero digit of the Boltzmann constant?",
            "el": "Ποιο είναι το πρώτο μη μηδενικό ψηφίο της σταθεράς Boltzmann;",
            "zh": "玻尔兹曼常数的第一个非零数字是多少？",
            "tr": "Boltzmann sabitinin ilk sıfır olmayan basamağı nedir?",
            "sw": "Ni tarakimu gani ya kwanza isiyo sifuri ya konstanti ya Boltzmann?",
            "ka": "რა არის ბოლცმანის მუდმივის პირველი არანულოვანი ციფრი?"
        },
        10: {
            "en": "What is the first digit of the gas constant?",
            "el": "Ποιο είναι το πρώτο ψηφίο της σταθεράς των αερίων;",
            "zh": "气体常数的第一位数字是多少？",
            "tr": "Gaz sabitinin ilk basamağı nedir?",
            "sw": "Ni tarakimu gani ya kwanza ya konstanti ya gesi?",
            "ka": "რა არის გაზის მუდმივის პირველი ციფრი?"
        },
        11: {
            "en": "What is the value of i^2?",
            "el": "Ποια είναι η τιμή του i^2;",
            "zh": "i^2 的值是多少？",
            "tr": "i^2 değeri nedir?",
            "sw": "Thamani ya i^2 ni nini?",
            "ka": "რა არის i^2-ის მნიშვნელობა?"
        },
        12: {
            "en": "What is the first digit of the square root of 2?",
            "el": "Ποιο είναι το πρώτο ψηφίο της τετραγωνικής ρίζας του 2;",
            "zh": "√2 的第一位数字是多少？",
            "tr": "√2 sayısının ilk basamağı nedir?",
            "sw": "Ni tarakimu gani ya kwanza ya mzizi wa pili wa 2?",
            "ka": "რა არის √2-ის პირველი ციფრი?"
        },
        13: {
            "en": "What is the value of infinity?",
            "el": "Ποια είναι η τιμή του απείρου;",
            "zh": "无穷大的值是多少？",
            "tr": "Sonsuzluğun değeri nedir?",
            "sw": "Thamani ya ukomo (infinity) ni nini?",
            "ka": "რა არის უსასრულობის მნიშვნელობა?"
        },
        14: {
            "en": "What is the first non-zero digit of vacuum electric permittivity?",
            "el": "Ποιο είναι το πρώτο μη μηδενικό ψηφίο της ηλεκτρικής διαπερατότητας του κενού;",
            "zh": "真空介电常数的第一个非零数字是多少？",
            "tr": "Boşluğun elektriksel geçirgenliğinin ilk sıfır olmayan basamağı nedir?",
            "sw": "Ni tarakimu gani ya kwanza isiyo sifuri ya upenyezaji wa umeme wa ombwe?",
            "ka": "რა არის ვაკუუმის ელექტრული პერმიტივობის პირველი არანულოვანი ციფრი?"
        },
        15: {
            "en": "What is the absolute value of zero?",
            "el": "Ποια είναι η απόλυτη τιμή του μηδενός;",
            "zh": "零的绝对值是多少？",
            "tr": "Sıfırın mutlak değeri nedir?",
            "sw": "Thamani halisi ya sifuri ni nini?",
            "ka": "რა არის ნულის აბსოლუტური მნიშვნელობა?"
        }
}
QUESTION_CONFIG["level2"] = {
    1: {
        "en": "What is pi multiplied by 3?",
        "el": "Ποιο είναι το π επί 3;",
        "zh": "π 乘以 3 等于多少？",
        "tr": "Pi sayısı 3 ile çarpıldığında sonuç nedir?",
        "sw": "Pi ikizidishwa kwa 3 ni nini?",
        "ka": "რა არის π გამრავლებული 3-ზე?"
    },
    2: {
        "en": "What is e^2?",
        "el": "Ποια είναι η τιμή του e^2;",
        "zh": "e^2 等于多少？",
        "tr": "e^2 değeri nedir?",
        "sw": "Thamani ya e^2 ni nini?",
        "ka": "რა არის e^2-ის მნიშვნელობა?"
    },
    3: {
        "en": "What is 5*phi-2?",
        "el": "Ποιο είναι το 5*φ-2;",
        "zh": "5×φ−2 等于多少？",
        "tr": "5*phi-2 işleminin sonucu nedir?",
        "sw": "5*phi-2 ni nini?",
        "ka": "რა არის 5*φ-2?"
    },
    4: {
        "en": "How much time (in seconds) does it take light to travel a distance of 100 million kilometers?",
        "el": "Πόσος χρόνος (σε δευτερόλεπτα) χρειάζεται το φως για να διανύσει 100 εκατομμύρια χιλιόμετρα;",
        "zh": "光传播一亿公里需要多少秒？",
        "tr": "Işık 100 milyon kilometreyi kaç saniyede alır?",
        "sw": "Inachukua muda gani (sekunde) kwa mwanga kusafiri kilomita milioni 100?",
        "ka": "რამდენი დრო (წამებში) სჭირდება სინათლეს 100 მილიონი კილომეტრის გასავლელად?"
    },
    5: {
        "en": "What is the gravitational constant multiplied by 7?",
        "el": "Ποια είναι η βαρυτική σταθερά επί 7;",
        "zh": "引力常数乘以 7 等于多少？",
        "tr": "Yerçekimi sabiti 7 ile çarpılırsa sonuç nedir?",
        "sw": "Konstanti ya mvuto ikizidishwa kwa 7 ni nini?",
        "ka": "რა არის გრავიტაციის მუდმივა გამრავლებული 7-ზე?"
    },
    6: {
        "en": "If the frequency of a photon is 4 Hz, what is its energy? Use the formula E=h*ν.",
        "el": "Αν η συχνότητα ενός φωτονίου είναι 4 Hz, ποια είναι η ενέργειά του; Χρησιμοποιήστε τον τύπο E=h*ν.",
        "zh": "如果光子的频率是 4 Hz，它的能量是多少？使用公式 E=hν。",
        "tr": "Bir fotonun frekansı 4 Hz ise enerjisi nedir? E=h*ν formülünü kullanın.",
        "sw": "Ikiwa mzunguko wa fotoni ni 4 Hz, nishati yake ni nini? Tumia fomula E=h*ν.",
        "ka": "თუ ფოტონის სიხშირე არის 4 Hz, რა არის მისი ენერგია? გამოიყენეთ ფორმულა E=h*ν."
    },
    7: {
        "en": "If an electron has a charge of −e, what is the charge of two electrons?",
        "el": "Αν ένα ηλεκτρόνιο έχει φορτίο −e, ποιο είναι το φορτίο δύο ηλεκτρονίων;",
        "zh": "如果一个电子的电荷是 −e，那么两个电子的电荷是多少？",
        "tr": "Bir elektronun yükü −e ise iki elektronun yükü nedir?",
        "sw": "Ikiwa elektroni moja ina chaji ya −e, chaji ya elektroni mbili ni nini?",
        "ka": "თუ ელექტრონის მუხტი არის −e, რა არის ორი ელექტრონის მუხტი?"
    },
    8: {
        "en": "How many atoms are there in 1 mole of any element?",
        "el": "Πόσα άτομα υπάρχουν σε 1 mol οποιουδήποτε στοιχείου;",
        "zh": "1 摩尔任何元素中有多少个原子？",
        "tr": "Herhangi bir elementin 1 molünde kaç atom vardır?",
        "sw": "Kuna atomi ngapi katika moli 1 ya elementi yoyote?",
        "ka": "რამდენი ატომია ნებისმიერი ელემენტის 1 მოლში?"
    },
    9: {
        "en": "Calculate the energy associated with a temperature of 300 K for a single particle using the formula E=kT.",
        "el": "Υπολογίστε την ενέργεια για θερμοκρασία 300 K για ένα σωματίδιο χρησιμοποιώντας τον τύπο E=kT.",
        "zh": "使用公式 E=kT，计算单个粒子在 300 K 时的能量。",
        "tr": "E=kT formülünü kullanarak 300 K sıcaklıktaki tek bir parçacığın enerjisini hesaplayın.",
        "sw": "Tumia fomula E=kT kuhesabu nishati ya chembe moja katika 300 K.",
        "ka": "გამოთვალეთ ერთ ნაწილაკზე 300 K ტემპერატურის შესაბამისი ენერგია ფორმულით E=kT."
    },
    10: {
        "en": "What is the gas constant divided by 2?",
        "el": "Ποια είναι η σταθερά των αερίων δια 2;",
        "zh": "气体常数除以 2 等于多少？",
        "tr": "Gaz sabiti 2'ye bölünürse sonuç nedir?",
        "sw": "Konstanti ya gesi ikigawanywa kwa 2 ni nini?",
        "ka": "რა არის გაზის მუდმივა გაყოფილი 2-ზე?"
    },
    11: {
        "en": "What is the value of i^3?",
        "el": "Ποια είναι η τιμή του i^3;",
        "zh": "i^3 的值是多少？",
        "tr": "i^3 değeri nedir?",
        "sw": "Thamani ya i^3 ni nini?",
        "ka": "რა არის i^3-ის მნიშვნელობა?"
    },
    12: {
        "en": "What is the square root of 2 multiplied by 3 (approximately)?",
        "el": "Ποια είναι περίπου η τετραγωνική ρίζα του 2 επί 3;",
        "zh": "√2 乘以 3 约等于多少？",
        "tr": "√2 sayısı 3 ile çarpıldığında yaklaşık değer nedir?",
        "sw": "√2 ikizidishwa kwa 3 ni takriban nini?",
        "ka": "რა არის √2 გამრავლებული 3-ზე (დაახლოებით)?"
    },
    13: {
        "en": "What is the limit of 1/x as x approaches infinity?",
        "el": "Ποιο είναι το όριο του 1/x καθώς x → άπειρο;",
        "zh": "当 x 趋于无穷大时，1/x 的极限是多少？",
        "tr": "x sonsuza giderken 1/x'in limiti nedir?",
        "sw": "Kikomo cha 1/x ni nini x inapokaribia ukomo?",
        "ka": "რა არის 1/x-ის ზღვარი როცა x უსასრულობას უახლოვდება?"
    },
    14: {
        "en": "If you add the value of vacuum electric permittivity to itself, what do you get?",
        "el": "Αν προσθέσετε τη διαπερατότητα του κενού με τον εαυτό της, τι προκύπτει;",
        "zh": "如果将真空介电常数加到自身，会得到什么？",
        "tr": "Boşluğun elektriksel geçirgenliğini kendisiyle toplarsanız sonuç nedir?",
        "sw": "Ukiongeza upenyezaji wa umeme wa ombwe kwa wenyewe, unapata nini?",
        "ka": "თუ ვაკუუმის ელექტრულ პერმიტივობას თავის თავს დაუმატებთ, რას მიიღებთ?"
    },
    15: {
        "en": "What is 300 multiplied by zero?",
        "el": "Ποιο είναι το 300 επί μηδέν;",
        "zh": "300 乘以 0 等于多少？",
        "tr": "300 sıfır ile çarpılırsa sonuç nedir?",
        "sw": "300 ikizidishwa kwa sifuri ni nini?",
        "ka": "რა არის 300 გამრავლებული ნულზე?"
    }
}
QUESTION_CONFIG["level3"] = {
    1: {
        "en": "What is the Earth's surface area?",
        "el": "Ποια είναι η επιφάνεια της Γης;",
        "zh": "地球的表面积是多少？",
        "tr": "Dünya'nın yüzey alanı nedir?",
        "sw": "Eneo la uso wa Dunia ni kiasi gani?",
        "ka": "რა არის დედამიწის ზედაპირის ფართობი?"
    },
    2: {
        "en": "If a population grows continuously at a rate of 5% per year, by what factor will it increase in 10 years?",
        "el": "Αν ένας πληθυσμός αυξάνεται συνεχώς με 5% ετησίως, κατά πόσο θα αυξηθεί σε 10 χρόνια;",
        "zh": "如果一个种群以每年 5% 的速率连续增长，10 年后将增长多少倍？",
        "tr": "Bir popülasyon yılda %5 oranında sürekli büyürse 10 yılda kaç kat artar?",
        "sw": "Ikiwa idadi ya watu inaongezeka kwa 5% kwa mwaka mfululizo, itaongezeka kwa kiwango gani baada ya miaka 10?",
        "ka": "თუ მოსახლეობა წლიურად 5%-ით უწყვეტად იზრდება, რამდენჯერ გაიზრდება 10 წელში?"
    },
    3: {
        "en": "If a rectangle has sides in the golden ratio and the longer side is 8 cm, what is the length of the other side?",
        "el": "Αν ένα ορθογώνιο έχει πλευρές σε χρυσή τομή και η μεγάλη πλευρά είναι 8 cm, ποια είναι η άλλη πλευρά;",
        "zh": "如果一个矩形的边长满足黄金比例，且长边为 8 cm，短边是多少？",
        "tr": "Bir dikdörtgenin kenarları altın orandaysa ve uzun kenar 8 cm ise diğer kenar nedir?",
        "sw": "Ikiwa mstatili una uwiano wa dhahabu na upande mrefu ni 8 cm, upande mwingine ni gani?",
        "ka": "თუ მართკუთხედის გვერდები ოქროს შეფარდებაშია და დიდი გვერდი 8 სმ-ია, მეორე გვერდი რა არის?"
    },
    4: {
        "en": "What is the energy equivalent of 8 grams of mass?",
        "el": "Ποια είναι η ενέργεια που αντιστοιχεί σε μάζα 8 γραμμαρίων;",
        "zh": "8 克质量对应的能量是多少？",
        "tr": "8 gram kütlenin enerji eşdeğeri nedir?",
        "sw": "Nishati inayolingana na gramu 8 za wingi ni nini?",
        "ka": "რა არის 8 გრამი მასის ენერგეტიკული ეკვივალენტი?"
    },
    5: {
        "en": "If two 15 kg masses are placed 2 meters apart, calculate the gravitational force between them.",
        "el": "Αν δύο μάζες 15 kg απέχουν 2 μέτρα, υπολογίστε τη βαρυτική δύναμη μεταξύ τους.",
        "zh": "如果两个 15 kg 的物体相距 2 米，计算它们之间的引力。",
        "tr": "İki 15 kg kütle 2 metre arayla yerleştirilirse aralarındaki yerçekimi kuvveti nedir?",
        "sw": "Ikiwa vipimo viwili vya kilo 15 viko umbali wa mita 2, hesabu nguvu ya mvuto kati yao.",
        "ka": "თუ ორი 15 კგ მასა 2 მეტრითაა დაშორებული, გამოთვალეთ მათ შორის გრავიტაციული ძალა."
    },
    6: {
        "en": "If a metal has a work function of 4.5×10^(−19) J, what is the minimum frequency of light required to eject an electron?",
        "el": "Αν ένα μέταλλο έχει έργο εξόδου 4.5×10^(−19) J, ποια είναι η ελάχιστη συχνότητα φωτός για εκπομπή ηλεκτρονίου;",
        "zh": "如果金属的逸出功为 4.5×10^(−19) J，所需的最小光频率是多少？",
        "tr": "Bir metalin iş fonksiyonu 4.5×10^(−19) J ise elektronu koparmak için gereken minimum frekans nedir?",
        "sw": "Ikiwa metali ina kazi ya kutoa elektroni ya 4.5×10^(−19) J, ni mzunguko gani mdogo wa mwanga unaohitajika?",
        "ka": "თუ მეტალს აქვს გამოსვლის მუშაობა 4.5×10^(−19) J, რა არის მინიმალური სიხშირე ელექტრონის გამოსაგდებად?"
    },
    7: {
        "en": "A capacitor stores a charge of 3.2×10^(−18) C. How many elementary charges is this?",
        "el": "Ένας πυκνωτής αποθηκεύει φορτίο 3.2×10^(−18) C. Πόσα στοιχειώδη φορτία είναι αυτά;",
        "zh": "一个电容器存储 3.2×10^(−18) C 的电荷，这相当于多少个基本电荷？",
        "tr": "Bir kondansatör 3.2×10^(−18) C yük depoluyorsa bu kaç temel yüke eşittir?",
        "sw": "Kondensa ina chaji ya 3.2×10^(−18) C. Hii ni sawa na chaji ngapi za msingi?",
        "ka": "კონდენსატორი ინახავს 3.2×10^(−18) C მუხტს. რამდენ ელემენტარულ მუხტს უდრის ეს?"
    },
    8: {
        "en": "Calculate the number of molecules in 54 grams of water.",
        "el": "Υπολογίστε τον αριθμό μορίων σε 54 γραμμάρια νερού.",
        "zh": "计算 54 克水中分子的数量。",
        "tr": "54 gram sudaki molekül sayısını hesaplayın.",
        "sw": "Hesabu idadi ya molekuli katika gramu 54 za maji.",
        "ka": "გამოთვალეთ 54 გრამ წყალში არსებული მოლეკულების რაოდენობა."
    },
    9: {
        "en": "What is the temperature at which the average kinetic energy is 1.9×10^(−21) J?",
        "el": "Σε ποια θερμοκρασία η μέση κινητική ενέργεια είναι 1.9×10^(−21) J;",
        "zh": "当平均动能为 1.9×10^(−21) J 时温度是多少？",
        "tr": "Ortalama kinetik enerji 1.9×10^(−21) J ise sıcaklık nedir?",
        "sw": "Ni joto gani ambapo nishati ya wastani ya kinetiki ni 1.9×10^(−21) J?",
        "ka": "რა ტემპერატურაზე არის საშუალო კინეტიკური ენერგია 1.9×10^(−21) J?"
    },
    10: {
        "en": "If you have 2 moles of an ideal gas at 300 K, what is the pressure if volume is 10 liters?",
        "el": "Αν έχετε 2 mol ιδανικού αερίου στους 300 K, ποια είναι η πίεση αν ο όγκος είναι 10 λίτρα;",
        "zh": "如果有 2 摩尔理想气体在 300 K，体积为 10 升，压力是多少？",
        "tr": "300 K sıcaklıkta 2 mol ideal gazın hacmi 10 litre ise basınç nedir?",
        "sw": "Ikiwa una moli 2 za gesi bora katika 300 K na ujazo ni lita 10, shinikizo ni nini?",
        "ka": "თუ გაქვთ 2 მოლი იდეალური გაზი 300 K ტემპერატურაზე და მოცულობა არის 10 ლიტრი, რა არის წნევა?"
    },
    11: {
        "en": "If z1=1+i and z2=1−i, what is z1*z2?",
        "el": "Αν z1=1+i και z2=1−i, ποιο είναι το γινόμενο z1*z2;",
        "zh": "如果 z1=1+i 且 z2=1−i，z1*z2 等于多少？",
        "tr": "z1=1+i ve z2=1−i ise z1*z2 nedir?",
        "sw": "Ikiwa z1=1+i na z2=1−i, z1*z2 ni nini?",
        "ka": "თუ z1=1+i და z2=1−i, რა არის z1*z2?"
    },
    12: {
        "en": "If a square has side 5, what is the diagonal?",
        "el": "Αν ένα τετράγωνο έχει πλευρά 5, ποια είναι η διαγώνιος;",
        "zh": "如果正方形边长为 5，对角线是多少？",
        "tr": "Bir karenin kenarı 5 ise köşegen nedir?",
        "sw": "Ikiwa mraba una upande wa 5, diagonal ni nini?",
        "ka": "თუ კვადრატის გვერდი არის 5, რა არის დიაგონალი?"
    },
    13: {
        "en": "What is the horizontal asymptote of f(x)=(5x+30000)/(x+1000)?",
        "el": "Ποια είναι η οριζόντια ασύμπτωτη της f(x)=(5x+30000)/(x+1000);",
        "zh": "函数 f(x)=(5x+30000)/(x+1000) 的水平渐近线是什么？",
        "tr": "f(x)=(5x+30000)/(x+1000) fonksiyonunun yatay asimptotu nedir?",
        "sw": "Asimptoti ya mlalo ya f(x)=(5x+30000)/(x+1000) ni nini?",
        "ka": "რა არის ფუნქციის f(x)=(5x+30000)/(x+1000) ჰორიზონტალური ასიმპტოტა?"
    },
    14: {
        "en": "Calculate the electric force between charges 3μC and 5μC at 12 m.",
        "el": "Υπολογίστε τη δύναμη μεταξύ φορτίων 3μC και 5μC σε απόσταση 12 m.",
        "zh": "计算相距 12 m 的 3μC 和 5μC 电荷之间的电力。",
        "tr": "12 m uzaklıktaki 3μC ve 5μC yükler arasındaki elektrik kuvvetini hesaplayın.",
        "sw": "Hesabu nguvu ya umeme kati ya chaji 3μC na 5μC umbali wa mita 12.",
        "ka": "გამოთვალეთ ელექტრული ძალა 3μC და 5μC მუხტებს შორის 12 მეტრზე."
    },
    15: {
        "en": "What is the limit of sin(x)/x as x approaches 0?",
        "el": "Ποιο είναι το όριο του sin(x)/x καθώς x → 0;",
        "zh": "当 x 趋于 0 时，sin(x)/x 的极限是多少？",
        "tr": "x sıfıra yaklaşırken sin(x)/x'in limiti nedir?",
        "sw": "Kikomo cha sin(x)/x ni nini x inapokaribia 0?",
        "ka": "რა არის sin(x)/x-ის ზღვარი როცა x ნულს უახლოვდება?"
    }
}
def build_prompt(worldA_text, worldB_text, question, target_world, lang):
    cfg = LANG_CONFIG[lang]
    return f"""{worldA_text}\n\n{worldB_text}\n\n{cfg["answer"]}\n\nQuestion: {question} in {target_world}\n\nInstructions:\n{cfg["instruction"]}\n{cfg["return_line"]}\n\n{cfg["format"]}\n{cfg["final"]}"""

df_level1 = pd.read_excel('constants.xlsx', sheet_name='level1', engine='openpyxl')
df_level2 = pd.read_excel('constants.xlsx', sheet_name='level2', engine='openpyxl')
df_level3 = pd.read_excel('constants.xlsx', sheet_name='level3', engine='openpyxl')

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
    print(f"\n{'='*30}\n🚀 Starting Level {level_num}\n{'='*30}")

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
                            # Μικρό max_new_tokens για ταχύτητα αφού θέλουμε μόνο έναν αριθμό
                            output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

                        raw_output = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

                        # Προσθήκη στη λίστα
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

        output_csv = os.path.join(output_dir, f"qwen72_lang_con_zs_{level_num}{variant_name}.csv")
        pd.DataFrame(current_results).to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

print("All runs completed.")
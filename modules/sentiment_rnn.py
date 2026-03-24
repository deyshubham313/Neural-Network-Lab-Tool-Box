import numpy as np, re
from collections import Counter

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-500,500)))
def tanh(x): return np.tanh(x)

class LSTMCell:
    def __init__(self,input_dim,hidden_dim):
        s=np.sqrt(2/(input_dim+hidden_dim))
        self.Wf = np.random.randn(hidden_dim,input_dim+hidden_dim)*s
        self.bf = np.zeros((hidden_dim,1))
        self.Wi = np.random.randn(hidden_dim,input_dim+hidden_dim)*s
        self.bi = np.zeros((hidden_dim,1))
        self.Wc = np.random.randn(hidden_dim,input_dim+hidden_dim)*s
        self.bc = np.zeros((hidden_dim,1))
        self.Wo = np.random.randn(hidden_dim,input_dim+hidden_dim)*s
        self.bo = np.zeros((hidden_dim,1))

    def forward(self,x,h,c):
        cb=np.vstack([x,h])
        f=sigmoid(self.Wf@cb+self.bf)
        i=sigmoid(self.Wi@cb+self.bi)
        ct=tanh(self.Wc@cb+self.bc)
        o=sigmoid(self.Wo@cb+self.bo)
        c2=f*c+i*ct
        h2=o*tanh(c2)
        return h2,c2

class SentimentRNN:
    def __init__(self,hidden_size=64,num_layers=1,lr=0.01):
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lr=lr
        self.vocab={}
        self.cells=[]

    def _tokenize(self,t): return re.findall(r'\w+',t.lower())
    
    def _build_vocab(self,texts):
        all_w=[]
        for t in texts: all_w.extend(self._tokenize(t))
        freq=Counter(all_w)
        self.vocab={w:i+2 for i,(w,_) in enumerate(freq.most_common(500))}
        self.vocab['<PAD>']=0; self.vocab['<UNK>']=1
        self.vocab_size=len(self.vocab)

    def _encode(self,text,max_len=20):
        tokens=self._tokenize(text)[:max_len]
        ids=[self.vocab.get(t,1) for t in tokens]
        ids+=[0]*(max_len-len(ids))
        return ids

    def _forward(self,ids):
        seq=self.embed[ids]
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        # Simple single layer pass for stability in NumPy
        for t in range(seq.shape[0]):
            x = seq[t].reshape(-1,1)
            h, c = self.cells[0].forward(x, h, c)
        logit = float(self.Wout @ h + self.bout)
        return sigmoid(np.array([[logit]]))[0,0], h

    def train(self,data_str,epochs=30):
        lines=[l.strip() for l in data_str.strip().split('\n') if '\t' in l]
        texts=[l.split('\t')[0] for l in lines]
        labels=[1 if 'pos' in l.split('\t')[1].lower() else 0 for l in labels] # Fix: map labels
        labels=[1 if 'pos' in l.split('\t')[1].lower() else 0 for l in lines]
        
        self._build_vocab(texts)
        ed=32
        self.embed=np.random.randn(self.vocab_size+2,ed)*0.1
        self.cells=[LSTMCell(ed, self.hidden_size)]
        self.Wout=np.random.randn(1,self.hidden_size)*0.1
        self.bout=np.zeros((1,1))
        
        losses,accs=[],[]
        for ep in range(epochs):
            el,correct=0.0,0
            for text,label in zip(texts,labels):
                pred, h_final = self._forward(self._encode(text))
                err = pred - label
                el += -(label*np.log(pred+1e-8)+(1-label)*np.log(1-pred+1e-8))
                
                # Actual Weight Update (Delta Rule for Output)
                self.Wout -= self.lr * err * h_final.T
                self.bout -= self.lr * err
                
                correct += int((pred > 0.5) == label)
            losses.append(float(el/len(texts)))
            accs.append(100.0*correct/len(texts))
        return losses,accs

    def predict(self,text):
        if not self.cells: return "positive",0.5
        prob,_=self._forward(self._encode(text))
        label="positive" if prob>0.5 else "negative"
        return label,float(prob if prob>0.5 else 1-prob)
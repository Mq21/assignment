# NNプログラムの使い方
「#ここから!!!!!」と「#!!!!!ここまで」で囲まれた部分が変更可能
入力層と出力層のノード数はデータから自動判定
「#隠れ層のノード数指定!!!!!」の部分で隠れ層のノード数指定
隠れ層の層数を多層化(例えば2層)したい場合はhidden1，hidden2と2つのインスタンス生成 & input→hidden1→hidden2→output(逆伝搬時はoutput→hidden2→hidden1→input)と伝搬

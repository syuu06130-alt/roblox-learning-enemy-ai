--[[
	NeuralNetwork.lua
	-------------------------------------------------------------------
	Roblox上で完結する超軽量フィードフォワード・ニューラルネットワーク。

	構成: 入力層 -> 隠れ層(1層) -> 出力層
	活性化関数: 隠れ層 = tanh, 出力層 = tanh (-1〜1の範囲。移動方向に使いやすい)

	学習方式: オンライン強化学習(疑似Q学習 / 方策勾配の簡易版)
	  毎ステップ「今回出した行動」に対して「報酬」を計算し、
	  報酬が正なら今回の出力方向に重みを少し近づけ、
	  報酬が負なら今回の出力方向から重みを少し遠ざける。
	  これにより重い学習フレームワークなしで「プレイ中に少しずつ賢くなる」
	  挙動をLua単体・1フレーム内の軽い計算量で実現する。

	注意:
	  本物のディープラーニング(誤差逆伝播による多層学習)ではなく、
	  Robloxの実行時間制約内で動作する軽量な近似実装です。
	  それでも「入力->隠れ層->出力」の構造、重み行列、
	  報酬ベースのオンライン重み更新までを完全に自前実装しています。
--]]

local NeuralNetwork = {}
NeuralNetwork.__index = NeuralNetwork

-- ============================================================
-- ユーティリティ関数
-- ============================================================

local function tanh(x)
	-- Robloxのmath libraryにtanhが無い場合があるため自前定義
	if x > 20 then return 1 end
	if x < -20 then return -1 end
	local e2x = math.exp(2 * x)
	return (e2x - 1) / (e2x + 1)
end

local function tanhDerivative(y)
	-- yはtanh適用後の値。 d/dx tanh(x) = 1 - tanh(x)^2
	return 1 - (y * y)
end

local function randomWeight(rng)
	-- -1 〜 1 の範囲で初期化(Xavier風に軽くスケーリング)
	return (rng:NextNumber() * 2 - 1)
end

-- ============================================================
-- コンストラクタ
-- ============================================================

--[[
	NeuralNetwork.new(inputSize, hiddenSize, outputSize, seed)
	新しいネットワークを生成する(ランダム初期化)。
--]]
function NeuralNetwork.new(inputSize, hiddenSize, outputSize, seed)
	local self = setmetatable({}, NeuralNetwork)

	self.inputSize = inputSize
	self.hiddenSize = hiddenSize
	self.outputSize = outputSize

	local rng = Random.new(seed or os.clock() * 1000)

	-- 重み行列: W1 (input -> hidden), W2 (hidden -> output)
	self.W1 = {} -- [hiddenIndex][inputIndex]
	self.b1 = {} -- [hiddenIndex]
	self.W2 = {} -- [outputIndex][hiddenIndex]
	self.b2 = {} -- [outputIndex]

	for h = 1, hiddenSize do
		self.W1[h] = {}
		for i = 1, inputSize do
			self.W1[h][i] = randomWeight(rng) * 0.5
		end
		self.b1[h] = 0
	end

	for o = 1, outputSize do
		self.W2[o] = {}
		for h = 1, hiddenSize do
			self.W2[o][h] = randomWeight(rng) * 0.5
		end
		self.b2[o] = 0
	end

	-- 直近の順伝播のキャッシュ(学習時に使用)
	self._lastInput = nil
	self._lastHidden = nil
	self._lastOutput = nil

	return self
end

-- ============================================================
-- 順伝播 (Forward Propagation)
-- ============================================================

--[[
	network:Forward(inputArray) -> outputArray
	inputArrayは数値の配列。長さはinputSizeと一致させること。
--]]
function NeuralNetwork:Forward(inputArray)
	assert(#inputArray == self.inputSize, "入力サイズが一致しません")

	local hidden = {}
	for h = 1, self.hiddenSize do
		local sum = self.b1[h]
		local w1h = self.W1[h]
		for i = 1, self.inputSize do
			sum = sum + w1h[i] * inputArray[i]
		end
		hidden[h] = tanh(sum)
	end

	local output = {}
	for o = 1, self.outputSize do
		local sum = self.b2[o]
		local w2o = self.W2[o]
		for h = 1, self.hiddenSize do
			sum = sum + w2o[h] * hidden[h]
		end
		output[o] = tanh(sum)
	end

	-- 学習用にキャッシュ
	self._lastInput = inputArray
	self._lastHidden = hidden
	self._lastOutput = output

	return output
end

-- ============================================================
-- オンライン強化学習 (Reward-based Online Update)
-- ============================================================

--[[
	network:ReinforceLastAction(reward, learningRate)

	直前のForward呼び出しの出力に対して報酬を与え、重みを更新する。
	reward > 0  : その行動を強化(出力方向に重みを近づける)
	reward < 0  : その行動を抑制(出力方向から重みを遠ざける)

	これは厳密なバックプロパゲーションによる誤差最小化ではなく、
	「報酬を誤差信号として扱う」簡易版の方策勾配法(REINFORCEの軽量近似)。
	Robloxの1フレーム内で完結する計算量に収めるための設計。
--]]
function NeuralNetwork:ReinforceLastAction(reward, learningRate)
	if self._lastInput == nil then
		return -- まだ一度もForwardしていない
	end

	learningRate = learningRate or 0.05

	local input = self._lastInput
	local hidden = self._lastHidden
	local output = self._lastOutput

	-- 出力層の「疑似勾配」: 報酬 * (出力を強める方向)
	-- tanhの導関数を使い、出力が既に飽和(±1付近)している場合は更新を弱める
	local outputDelta = {}
	for o = 1, self.outputSize do
		outputDelta[o] = reward * tanhDerivative(output[o])
	end

	-- 隠れ層への逆伝播(誤差を1層分だけ伝える簡易版)
	local hiddenDelta = {}
	for h = 1, self.hiddenSize do
		local sum = 0
		for o = 1, self.outputSize do
			sum = sum + outputDelta[o] * self.W2[o][h]
		end
		hiddenDelta[h] = sum * tanhDerivative(hidden[h])
	end

	-- W2, b2 更新 (hidden -> output)
	for o = 1, self.outputSize do
		for h = 1, self.hiddenSize do
			self.W2[o][h] = self.W2[o][h] + learningRate * outputDelta[o] * hidden[h]
		end
		self.b2[o] = self.b2[o] + learningRate * outputDelta[o]
	end

	-- W1, b1 更新 (input -> hidden)
	for h = 1, self.hiddenSize do
		for i = 1, self.inputSize do
			self.W1[h][i] = self.W1[h][i] + learningRate * hiddenDelta[h] * input[i]
		end
		self.b1[h] = self.b1[h] + learningRate * hiddenDelta[h]
	end
end

-- ============================================================
-- シリアライズ (DataStore保存用)
-- ============================================================

function NeuralNetwork:Serialize()
	return {
		inputSize = self.inputSize,
		hiddenSize = self.hiddenSize,
		outputSize = self.outputSize,
		W1 = self.W1,
		b1 = self.b1,
		W2 = self.W2,
		b2 = self.b2,
	}
end

function NeuralNetwork.Deserialize(data)
	local self = setmetatable({}, NeuralNetwork)
	self.inputSize = data.inputSize
	self.hiddenSize = data.hiddenSize
	self.outputSize = data.outputSize
	self.W1 = data.W1
	self.b1 = data.b1
	self.W2 = data.W2
	self.b2 = data.b2
	self._lastInput = nil
	self._lastHidden = nil
	self._lastOutput = nil
	return self
end

--[[
	既存ネットワークの重みを少しだけ変異させたコピーを作る。
	(将来的に世代交代型へ拡張したい場合のためのユーティリティ。
	 現在のリアルタイム学習モードでは直接は使わないが、
	 個体が全く学習できなくなった場合のリセット手段として利用できる)
--]]
function NeuralNetwork:Mutate(amount, seed)
	local rng = Random.new(seed or os.clock() * 1000)
	for h = 1, self.hiddenSize do
		for i = 1, self.inputSize do
			self.W1[h][i] = self.W1[h][i] + (rng:NextNumber() * 2 - 1) * amount
		end
	end
	for o = 1, self.outputSize do
		for h = 1, self.hiddenSize do
			self.W2[o][h] = self.W2[o][h] + (rng:NextNumber() * 2 - 1) * amount
		end
	end
end

return NeuralNetwork

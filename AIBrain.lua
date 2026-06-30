--[[
	AIBrain.lua
	-------------------------------------------------------------------
	NPC1体分の「脳」を表すモジュール。

	役割:
	  1. 知覚: プレイヤーへの相対角度・距離、前方/左前/右前のレイキャストによる
	         壁までの距離を取得し、ニューラルネットへの入力ベクトルを作る。
	  2. 判断: NeuralNetwork:Forward() を呼び、移動方向(角度オフセット)と
	         速度係数を出力として得る。
	  3. 報酬: 直前の行動の結果(プレイヤーに近づけたか、壁にぶつかったか)を
	         評価して報酬を計算する。
	  4. 学習: 報酬をもとに NeuralNetwork:ReinforceLastAction() を呼び、
	         リアルタイムに重みを更新する(プレイ中に少しずつ賢くなる)。

	入力ベクトル(全6個、いずれも -1〜1 に正規化):
	  [1] プレイヤーへの相対角度 (左右どちらにどれだけ向きを変えるべきか)
	  [2] プレイヤーまでの距離 (正規化済み。遠いほど1に近い)
	  [3] 前方レイキャストの空き具合 (壁が近いほど-1、開けているほど1)
	  [4] 左前方レイキャストの空き具合
	  [5] 右前方レイキャストの空き具合
	  [6] 直前フレームで壁に衝突したか (0 or 1)

	出力ベクトル(全2個、-1〜1):
	  [1] 旋回方向 (負=左に曲がる、正=右に曲がる)
	  [2] 前進速度係数 (基本的に常に前進だが、壁が近いと自然に弱くなるよう学習される)
--]]

local NeuralNetwork = require(script.Parent:WaitForChild("NeuralNetwork"))

local AIBrain = {}
AIBrain.__index = AIBrain

-- ============================================================
-- 設定値
-- ============================================================

local INPUT_SIZE = 6
local HIDDEN_SIZE = 8
local OUTPUT_SIZE = 2

local RAY_DISTANCE = 12 -- レイキャストの最大検知距離(スタッド)
local RAY_SIDE_ANGLE = 35 -- 左右レイの角度(度)
local SENSE_RADIUS = 80 -- プレイヤーを感知する最大距離
local LEARNING_RATE = 0.06

-- 報酬の重みづけ(チューニングしやすいよう定数化)
local REWARD_APPROACH = 1.0      -- プレイヤーに近づいたときの基礎報酬
local REWARD_RETREAT = -1.0      -- プレイヤーから離れたときの基礎ペナルティ
local REWARD_WALL_HIT = -3.0     -- 壁に衝突したときの強いペナルティ
local REWARD_OPEN_PATH = 0.15    -- 開けた方向へ進めたボーナス(障害物回避の学習促進)

-- ============================================================
-- コンストラクタ
-- ============================================================

--[[
	AIBrain.new(humanoidRootPart, humanoid, savedWeights)
	savedWeights が渡された場合は DataStore から復元した重みを使用、
	無ければ新規ランダム初期化。
--]]
function AIBrain.new(rootPart, humanoid, savedWeights)
	local self = setmetatable({}, AIBrain)

	self.rootPart = rootPart
	self.humanoid = humanoid

	if savedWeights then
		self.net = NeuralNetwork.Deserialize(savedWeights)
	else
		self.net = NeuralNetwork.new(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
	end

	self.lastDistanceToTarget = nil
	self.lastHitWall = false

	return self
end

-- ============================================================
-- 知覚: レイキャストで前方の空き具合を測る
-- ============================================================

local function castRay(origin, direction, maxDist, ignoreList)
	local params = RaycastParams.new()
	params.FilterType = Enum.RaycastFilterType.Exclude
	params.FilterDescendantsInstances = ignoreList

	local result = workspace:Raycast(origin, direction.Unit * maxDist, params)
	if result then
		return result.Distance / maxDist -- 0(近い)〜1(遠い)
	else
		return 1 -- 何も無ければ最大距離扱い(完全に開けている)
	end
end

--[[
	0〜1の「空き具合」を -1〜1 の入力値に変換
	(0=壁すぐ近く -> -1、1=完全に開けている -> 1)
--]]
local function clearanceToInput(clearance)
	return clearance * 2 - 1
end

-- ============================================================
-- 入力ベクトルの構築
-- ============================================================

function AIBrain:BuildInputVector(targetPosition)
	local rootPart = self.rootPart
	local myPos = rootPart.Position
	local myCFrame = rootPart.CFrame

	-- ターゲット(プレイヤー)への相対角度・距離
	local toTarget = targetPosition - myPos
	local flatToTarget = Vector3.new(toTarget.X, 0, toTarget.Z)
	local distance = flatToTarget.Magnitude

	local forwardDir = myCFrame.LookVector
	local flatForward = Vector3.new(forwardDir.X, 0, forwardDir.Z)

	local angleToTarget = 0
	if flatToTarget.Magnitude > 0.01 then
		local dot = flatForward.Unit:Dot(flatToTarget.Unit)
		dot = math.clamp(dot, -1, 1)
		local angle = math.acos(dot)

		-- 左右どちらにあるかを外積のY成分で判定
		local cross = flatForward.Unit:Cross(flatToTarget.Unit)
		if cross.Y < 0 then
			angle = -angle
		end
		angleToTarget = angle / math.pi -- -1〜1に正規化
	end

	local normalizedDistance = math.clamp(distance / SENSE_RADIUS, 0, 1)
	-- 「近いほど良い」を学習しやすくするため距離はそのまま(0=近い,1=遠い)を渡す

	-- レイキャスト3方向(前方・左前・右前)
	local ignoreList = {rootPart.Parent}
	local origin = myPos + Vector3.new(0, 1, 0) -- 足元より少し上から

	local frontClear = castRay(origin, forwardDir, RAY_DISTANCE, ignoreList)

	local leftCFrame = CFrame.Angles(0, math.rad(RAY_SIDE_ANGLE), 0)
	local leftDir = leftCFrame:VectorToWorldSpace(forwardDir)
	local leftClear = castRay(origin, leftDir, RAY_DISTANCE, ignoreList)

	local rightCFrame = CFrame.Angles(0, math.rad(-RAY_SIDE_ANGLE), 0)
	local rightDir = rightCFrame:VectorToWorldSpace(forwardDir)
	local rightClear = castRay(origin, rightDir, RAY_DISTANCE, ignoreList)

	local hitWallFlag = self.lastHitWall and 1 or 0

	local input = {
		angleToTarget,                  -- [1]
		normalizedDistance,              -- [2]
		clearanceToInput(frontClear),    -- [3]
		clearanceToInput(leftClear),     -- [4]
		clearanceToInput(rightClear),    -- [5]
		hitWallFlag,                     -- [6]
	}

	-- 報酬計算用に保存
	self._currentDistance = distance
	self._currentFrontClear = frontClear

	return input
end

-- ============================================================
-- 行動決定 + 移動実行
-- ============================================================

--[[
	AIBrain:Step(targetPosition, dt)
	1フレーム分の「知覚 -> 判断 -> 行動 -> 学習」を実行する。
--]]
function AIBrain:Step(targetPosition, dt)
	local rootPart = self.rootPart
	local humanoid = self.humanoid

	if not rootPart or not rootPart.Parent or not humanoid or humanoid.Health <= 0 then
		return
	end

	-- 1. 知覚
	local input = self:BuildInputVector(targetPosition)

	-- 2. 判断(ニューラルネット順伝播)
	local output = self.net:Forward(input)
	local turnAmount = output[1]      -- -1(左)〜1(右)
	local speedFactor = (output[2] + 1) / 2 -- 0〜1に変換(常に前進寄り)

	-- 3. 行動実行
	local currentCFrame = rootPart.CFrame
	local maxTurnPerStep = math.rad(120) * dt -- 1秒あたり最大120度まで旋回可能
	local newCFrame = currentCFrame * CFrame.Angles(0, turnAmount * maxTurnPerStep, 0)

	humanoid.WalkSpeed = 8 + speedFactor * 8 -- 8〜16の範囲で速度変化
	rootPart.CFrame = CFrame.new(rootPart.Position, rootPart.Position + newCFrame.LookVector)

	humanoid:Move(newCFrame.LookVector, false)

	-- 4. 報酬計算 + オンライン学習
	self:CalculateRewardAndLearn()
end

-- ============================================================
-- 報酬計算とオンライン学習
-- ============================================================

function AIBrain:CalculateRewardAndLearn()
	local reward = 0

	-- (a) プレイヤーへの接近/離脱
	if self.lastDistanceToTarget ~= nil then
		local delta = self.lastDistanceToTarget - self._currentDistance
		if delta > 0 then
			reward = reward + REWARD_APPROACH * math.clamp(delta, 0, 1)
		else
			reward = reward + REWARD_RETREAT * math.clamp(-delta, 0, 1)
		end
	end
	self.lastDistanceToTarget = self._currentDistance

	-- (b) 開けた方向へ進んでいるボーナス(障害物回避の学習を促進)
	if self._currentFrontClear then
		reward = reward + (self._currentFrontClear - 0.5) * REWARD_OPEN_PATH * 2
	end

	-- (c) 壁衝突ペナルティ
	if self.lastHitWall then
		reward = reward + REWARD_WALL_HIT
		self.lastHitWall = false -- フラグはリセット(OnTouchedで再セットされる)
	end

	-- ニューラルネットに報酬を渡してオンライン更新
	self.net:ReinforceLastAction(reward, LEARNING_RATE)
end

--[[
	外部(Touchedイベントなど)から「壁にぶつかった」と通知するための関数
--]]
function AIBrain:NotifyWallHit()
	self.lastHitWall = true
end

-- ============================================================
-- 重みの取得(DataStore保存用)
-- ============================================================

function AIBrain:GetWeights()
	return self.net:Serialize()
end

return AIBrain

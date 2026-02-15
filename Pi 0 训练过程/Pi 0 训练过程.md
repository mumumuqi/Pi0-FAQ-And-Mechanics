## 🔬 Pi0 训练完整微观数据流

### 📊 数据流全景图

```
原始数据集 (LeRobot/RLDS)
    ↓
[DataLoader 静态预处理]
    ↓
Batch 数据
    ↓
[train_step 训练循环]
    ↓
[loss_fn 损失函数]
    ↓
[compute_loss 模型内部] ← 动态增强在这里
    ↓
[embed_prefix + embed_suffix]
    ↓
[Transformer 前向传播]
    ↓
[计算损失并反向传播]
```

---

### 🎯 第一阶段：DataLoader 静态预处理

**位置**: [data_loader.py:172-191](vscode-webview://0jvd0mqagsl2s1ik7vc8vfbet0m7s3710uo1v8o7ljqnpf8qq82v/src/openpi/training/data_loader.py#L172-L191)

**输入**: 原始数据集样本

```python
{
    "observation": {
        "images": {
            "base_0_rgb": uint8[h, w, 3],      # 原始图片 [0-255]
            "left_wrist_0_rgb": uint8[h, w, 3],
            "right_wrist_0_rgb": uint8[h, w, 3]
        },
        "state": float32[s]                     # 机器人状态
    },
    "action": float32[ah, ad],                  # 动作序列
    "prompt": str                               # 文本指令
}
```

**Transform 管道** (按顺序执行):

1. **RepackTransform** [transforms.py:80-101]
    
    - 重组字典结构，统一命名
2. **DeltaActions** [transforms.py:204-222]
    
    - 将绝对动作转换为相对动作（如果配置启用）
    
    ```python
    actions[..., :dims] -= state[..., :dims]  # 相对于当前状态的增量
    ```
    
3. **ResizeImages** [transforms.py:185-191]
    
    - 缩放图片到 224×224（保持长宽比，填充黑边）
    
    ```python
    image = image_tools.resize_with_pad(v, 224, 224)
    ```
    
4. **TokenizePrompt** [transforms.py:248-266]
    
    - 将文本指令转换为 token IDs
    
    ```python
    tokens, token_masks = tokenizer.tokenize(prompt, state)
    # tokens: int32[l]  (l=48)
    # token_masks: bool[l]
    ```
    
5. **Normalize** [transforms.py:115-145]
    
    - 归一化状态和动作（z-score 或 quantile）
    
    ```python
    # z-score: (x - mean) / (std + 1e-6)
    # quantile: (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    ```
    
6. **PadStatesAndActions** [transforms.py:328-337]
    
    - 零填充到模型维度
    
    ```python
    state = pad_to_dim(state, model_action_dim, axis=-1)
    actions = pad_to_dim(actions, model_action_dim, axis=-1)
    ```

**输出**: 预处理后的 Batch

```python
{
    "image": {
        "base_0_rgb": float32[32, 224, 224, 3],      # [-1, 1]
        "left_wrist_0_rgb": float32[32, 224, 224, 3],
        "right_wrist_0_rgb": float32[32, 224, 224, 3]
    },
    "image_mask": {
        "base_0_rgb": bool[32],
        "left_wrist_0_rgb": bool[32],
        "right_wrist_0_rgb": bool[32]
    },
    "state": float32[32, 14],                        # 归一化后
    "tokenized_prompt": int32[32, 48],
    "tokenized_prompt_mask": bool[32, 48],
    "actions": float32[32, 50, 14]                   # 归一化后
}
```

---

### 🎯 第二阶段：train_step 训练循环

**位置**: [train.py:136-190](vscode-webview://0jvd0mqagsl2s1ik7vc8vfbet0m7s3710uo1v8o7ljqnpf8qq82v/scripts/train.py#L136-L190)

**输入**: 从 DataLoader 获取的 batch

```python
batch: tuple[Observation, Actions]
# Observation: 包含 images, image_masks, state, tokenized_prompt 等
# Actions: float32[32, 50, 14]
```

**核心代码**:

```python
def train_step(config, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()  # 设置为训练模式
    
    def loss_fn(model, rng, observation, actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)
    
    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    
    # 计算损失和梯度
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions
    )
    
    # 更新参数
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_state, info
```

---

### 🎯 第三阶段：compute_loss 模型内部

**位置**: [pi0.py:201-227](vscode-webview://0jvd0mqagsl2s1ik7vc8vfbet0m7s3710uo1v8o7ljqnpf8qq82v/src/openpi/models/pi0.py#L201-L227)

#### 步骤 1: 动态图像增强 (仅训练时)

**位置**: [pi0.py:206](vscode-webview://0jvd0mqagsl2s1ik7vc8vfbet0m7s3710uo1v8o7ljqnpf8qq82v/src/openpi/models/pi0.py#L206)

```python
preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
observation = _model.preprocess_observation(
    preprocess_rng, observation, train=True  # ← 关键！train=True
)
```

**preprocess_observation 内部** [model.py:186-294]:

```python
# 对每张图片进行增强
for key in image_keys:
    image = observation.images[key]  # float32[32, 224, 224, 3]
    
    if train:
        # 1. 转换到 [0, 1]
        image = image / 2.0 + 0.5
        
        # 2. 几何变换 (仅非手腕摄像头)
        if "wrist" not in key:
            transforms = [
                augmax.RandomCrop(int(224 * 0.95), int(224 * 0.95)),  # 裁剪 5%
                augmax.Resize(224, 224),                               # 拉伸回原尺寸
                augmax.Rotate((-5, 5)),                                # 旋转 ±5°
            ]
        
        # 3. 颜色变换 (所有摄像头)
        transforms += [
            augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
        ]
        
        # 4. 执行增强 (向量化)
        sub_rngs = jax.random.split(rng, image.shape[0])
        image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)
        
        # 5. 转换回 [-1, 1]
        image = image * 2.0 - 1.0
    
    out_images[key] = image
```

**输出**: 增强后的 observation

```python
observation.images: {
    "base_0_rgb": float32[32, 224, 224, 3],      # 已增强
    "left_wrist_0_rgb": float32[32, 224, 224, 3],
    "right_wrist_0_rgb": float32[32, 224, 224, 3]
}
```

#### 步骤 2: Flow Matching 噪声生成

**位置**: [pi0.py:208-213](vscode-webview://0jvd0mqagsl2s1ik7vc8vfbet0m7s3710uo1v8o7ljqnpf8qq82v/src/openpi/models/pi0.py#L208-L213)

```python
batch_shape = actions.shape[:-2]  # [32]
noise = jax.random.normal(noise_rng, actions.shape)  # float32[32, 50, 14]
time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001  # float32[32]

# 插值生成噪声动作
time_expanded = time[:, None, None]  # [32, 1, 1]
x_t = time_expanded * noise + (1 - time_expanded) * actions  # float32[32, 50, 14]
u_t = noise - actions  # 目标速度场
```

**数据含义**:

- `noise`: 纯随机噪声 (标准正态分布)
- `time`: 扩散时间步 ∈ (0.001, 1.0)
- `x_t`: 时间 t 的噪声动作 (插值)
- `u_t`: 从 x_t 到真实动作的速度场 (目标)

---

### 🎯 第四阶段：Embedding 生成

#### 4.1 Prefix Embedding (上下文)

**位置**: [pi0.py:133-153](vscode-webview://0jvd0mqagsl2s1ik7vc8vfbet0m7s3710uo1v8o7ljqnpf8qq82v/src/openpi/models/pi0.py#L133-L153)

```python
def embed_prefix(self, obs: _model.Observation):
    tokens = []
    input_mask = []
    ar_mask = []
    
    # 1. 图像 tokens (3 个摄像头)
    for key in _model.IMAGE_KEYS:
        img_tokens = self.PaliGemma.vit(obs.images[key])  # [32, 256, 2048]
        tokens.append(img_tokens)
        input_mask.append(jnp.ones((32, 256), dtype=jnp.bool_))
        ar_mask += [False] * 256  # 图像不参与自回归
    
    # 2. 语言 tokens
    if obs.tokenized_prompt is not None:
        lang_tokens = self.PaliGemma.llm.embedder(obs.tokenized_prompt)  # [32, 48, 2048]
        tokens.append(lang_tokens)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * 48
    
    # 3. 状态 token (仅 Pi0)
    if not self.pi05:
        state_token = self.state_proj(obs.state)[:, None, :]  # [32, 1, 2048]
        tokens.append(state_token)
        input_mask.append(jnp.ones((32, 1), dtype=jnp.bool_))
        ar_mask += [True]  # 状态参与自回归
    
    # 拼接
    prefix_tokens = jnp.concatenate(tokens, axis=1)  # [32, 817, 2048]
    prefix_mask = jnp.concatenate(input_mask, axis=1)  # [32, 817]
    prefix_ar_mask = jnp.array(ar_mask)  # [817]
    
    return prefix_tokens, prefix_mask, prefix_ar_mask
```

**输出形状**:

```python
prefix_tokens: float32[32, 817, 2048]  # 256*3 + 48 + 1 = 817
prefix_mask: bool[32, 817]
prefix_ar_mask: bool[817]
```

#### 4.2 Suffix Embedding (动作)

**位置**: [pi0.py:155-183](vscode-webview://0jvd0mqagsl2s1ik7vc8vfbet0m7s3710uo1v8o7ljqnpf8qq82v/src/openpi/models/pi0.py#L155-L183)

```python
def embed_suffix(self, obs, noisy_actions, timestep):
    tokens = []
    input_mask = []
    ar_mask = []
    
    # 1. 动作投影
    action_tokens = self.action_in_proj(noisy_actions)  # [32, 50, 2048]
    
    # 2. 时间嵌入 (Sine-Cosine)
    time_emb = posemb_sincos(timestep, 2048, min_period=4e-3, max_period=4.0)  # [32, 2048]
    
    if self.pi05:
        # Pi0.5: AdaRMS 条件
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        adarms_cond = time_emb
    else:
        # Pi0: 时间嵌入加到动作 tokens
        action_tokens = action_tokens + time_emb[:, None, :]
        adarms_cond = None
    
    tokens.append(action_tokens)
    input_mask.append(jnp.ones((32, 50), dtype=jnp.bool_))
    ar_mask += [True] * 50  # 动作全部参与自回归
    
    suffix_tokens = jnp.concatenate(tokens, axis=1)  # [32, 50, 2048]
    suffix_mask = jnp.concatenate(input_mask, axis=1)  # [32, 50]
    suffix_ar_mask = jnp.array(ar_mask)  # [50]
    
    return suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond
```

**输出形状**:

```python
suffix_tokens: float32[32, 50, 2048]
suffix_mask: bool[32, 50]
suffix_ar_mask: bool[50]
adarms_cond: float32[32, 2048] (仅 Pi0.5)
```

---

### 🎯 第五阶段：Transformer 前向传播

**位置**: [pi0.py:215-224](vscode-webview://0jvd0mqagsl2s1ik7vc8vfbet0m7s3710uo1v8o7ljqnpf8qq82v/src/openpi/models/pi0.py#L215-L224)

```python
# 1. 拼接 prefix + suffix
full_tokens = jnp.concatenate([prefix_tokens, suffix_tokens], axis=1)  # [32, 867, 2048]
full_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)  # [32, 867]

# 2. 生成注意力掩码
full_ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask])  # [867]
attn_mask = make_attn_mask(full_mask, full_ar_mask)  # [32, 867, 867]

# 3. 位置编码
positions = jnp.cumsum(full_mask, axis=1) - 1  # [32, 867]

# 4. Transformer 前向传播
if self.pi05:
    full_out = self.ActionExpert(full_tokens, mask=attn_mask, positions=positions, adarms_cond=adarms_cond)
else:
    full_out = self.PaliGemma.llm([full_tokens, None], mask=attn_mask, positions=positions)[0]

# 5. 提取动作输出
suffix_out = full_out[:, -self.action_horizon:]  # [32, 50, 2048]
```

**注意力掩码结构** [pi0.py:40-58]:

```python
def make_attn_mask(input_mask, ar_mask):
    # 因果掩码：只能看到过去
    attn_mask = ar_mask[None, :] <= ar_mask[:, None]  # [867, 867]
    # 有效掩码：排除 padding
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]  # [32, 867, 867]
    # 合并
    return jnp.logical_and(attn_mask, valid_mask)  # [32, 867, 867]
```

---

### 🎯 第六阶段：损失计算

**位置**: [pi0.py:226-227](vscode-webview://0jvd0mqagsl2s1ik7vc8vfbet0m7s3710uo1v8o7ljqnpf8qq82v/src/openpi/models/pi0.py#L226-L227)

```python
# 1. 预测速度场
v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])  # [32, 50, 14]

# 2. 计算 MSE 损失
loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [32, 50]

# 返回每个样本每个时间步的损失
return loss  # float32[32, 50]
```

**损失含义**:

- `v_t`: 模型预测的速度场 (从 x_t 到 x_0 的方向)
- `u_t`: 真实速度场 (noise - actions)
- 损失 = ||v_t - u_t||²

---

### 🎯 第七阶段：反向传播和参数更新

**位置**: [train.py:157-165](vscode-webview://0jvd0mqagsl2s1ik7vc8vfbet0m7s3710uo1v8o7ljqnpf8qq82v/scripts/train.py#L157-L165)
**第六阶段** 是在模型**内部**（`compute_loss` 函数），算出了具体相差多少。这个`compute_loss`函数是在train.py的def loss_fn调用的，

```python
# 1. 计算梯度
diff_state = nnx.DiffState(0, config.trainable_filter)
loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
    model, train_rng, observation, actions
)

# 2. 过滤可训练参数
params = state.params.filter(config.trainable_filter)

# 3. 优化器更新
updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
new_params = optax.apply_updates(params, updates)

# 4. 更新模型
nnx.update(model, new_params)
new_params = nnx.state(model)

# 5. 更新 EMA (如果启用)
if state.ema_decay is not None:
    new_ema_params = jax.tree.map(
        lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
        state.ema_params, new_params
    )
```
在 JAX/Flax (NNX) 框架中，这一步的代码写得非常显式（把底层的齿轮都露出来了）。结合笔记里 **【🎯 第七阶段：反向传播和参数更新】** 的代码，我们把它拆解成**“模型纠错五步曲”**：

### 第 1 步：拿到修改意见书 (计算梯度)

Python

```
# 1. 计算梯度
diff_state = nnx.DiffState(0, config.trainable_filter)
loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
    model, train_rng, observation, actions
)
```

- **在干嘛**：这是 JAX 的核心机制。`value_and_grad` 就像是一个极其严苛的监考老师。它不仅让你把题做完算出分数 (`loss`)，还会利用微积分的链式法则，帮你算出**“这道题做错，究竟是因为哪个脑细胞（参数）短路了”**，并把所有修改意见汇总成一份报告，也就是梯度 (`grads`)。
    
- **巧妙之处**：注意看 `diff_state` 和 `config.trainable_filter`。因为 Pi0 模型很大，我们不想（也没算力）去修改底座大模型 PaliGemma 的参数。这个过滤器就是告诉监考老师：“你只管给我算出**动作专家 (Action Expert)** 相关的错误就行了，其他的别动！”
    

### 第 2 步：把需要修的零件拆下来 (过滤可训练参数)

Python

```
# 2. 过滤可训练参数
params = state.params.filter(config.trainable_filter)
```

- **在干嘛**：拿到修改意见后，我们不能直接对着整辆车（整个大模型）乱敲乱打。这行代码的意思是，根据过滤规则，把那些**被允许修改的旧零件（旧权重）**单独从模型身上拆卸下来，赋值给 `params` 变量。
    

### 第 3 步：修理工执行操作 (优化器更新)

Python

```
# 3. 优化器更新
updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
new_params = optax.apply_updates(params, updates)
```

- **在干嘛**：这里登场的是**优化器 (Optimizer)**（比如常用的 AdamW 算法，代码里叫 `state.tx`）。
    
- **第一句**：修理工看着修改意见 (`grads`)，结合它自己过去的维修经验 (`state.opt_state`，比如动量信息，防止这次改得太猛)，算出了一个**完美的调整幅度 (`updates`)**。
    
- **第二句**：把这个调整幅度加到刚才拆下来的旧零件 (`params`) 上，诞生了**全新的、更聪明的零件 (`new_params`)**。
    

### 第 4 步：把新零件装回车上 (更新模型)

Python

```
# 4. 更新模型
nnx.update(model, new_params)
new_params = nnx.state(model)
```

- **在干嘛**：刚才新造出来的零件 (`new_params`) 还在外面放着。`nnx.update` 就是把它们咔哒一声，严丝合缝地重新安装回 `model` 这个大骨架里。至此，模型完成了这一次的自我进化！
    

### 第 5 步：留下一个稳重的“影子” (更新 EMA)

Python

```
# 5. 更新 EMA (如果启用)
if state.ema_decay is not None:
    new_ema_params = jax.tree.map(
        lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
        state.ema_params, new_params
    )
```

- **在干嘛**：模型每次更新就像年轻人在试错，步子迈得可能比较跳跃。为了在测试机器人时表现更稳定，代码维护了一个 **EMA (指数移动平均)** 版本的“影子参数”。
    
- **怎么算**：它会保留 99% 的老影子 (`old`)，只吸收 1% 刚出炉的新参数 (`new`)。这个影子模型学得很慢，但也极度平滑，不会因为某一个 Batch 的极端数据而突然动作抽搐。
    

---

**一句话总结这五步：**

发现错误算梯度 $\rightarrow$ 拆下旧参数 $\rightarrow$ 结合梯度算出新参数 $\rightarrow$ 把新参数装进模型 $\rightarrow$ 顺手更新一下平滑备份。
import re, sys, os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler

current_file = Path(__file__)
sys.path.append(str(current_file.parent))
from hub_mixin import CompatiblePyTorchModelHubMixin
from rdt.model import RDT
from rdt.qformer import LanguageGuidedQFormer


class LangFiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioned on language.

    Pools valid language tokens into a single vector, then produces
    per-channel (γ, β) to scale-and-shift the state/action token sequence:
        out = γ(lang) * x + β(lang)

    Initialized near identity (γ≈1, β≈0) so that fine-tuning from a
    pretrained checkpoint starts with zero perturbation.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gamma_proj = nn.Linear(hidden_size, hidden_size)
        self.beta_proj  = nn.Linear(hidden_size, hidden_size)
        # identity initialisation for stable continued training
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, T, H) state/action tokens
        lang: (B, H)    pooled language embedding
        """
        gamma = self.gamma_proj(lang).unsqueeze(1)  # (B, 1, H)
        beta  = self.beta_proj(lang).unsqueeze(1)   # (B, 1, H)
        return x * gamma + beta


class ArmOrderPrefix(nn.Module):
    """Three learned tokens (one per block-grasp step) in hidden_size space,
    prepended to the language cross-attention K/V sequence.

    Unlike language tokens whose arm-order signal must survive T5 + lang_adaptor
    compression, these tokens carry arm-order information DIRECTLY into every
    RDT cross-attention block with no intermediate processing loss.

    Encoding: 0=left arm, 1=right arm, 2=unknown (zero vector, acts as no-op).
    At inference with unseen instructions that lack arm info, all-unknown tokens
    are zero vectors and have negligible effect on cross-attention.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        # 3 possible values per step
        self.embedding = nn.Embedding(3, hidden_size)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.embedding.weight[2])   # unknown slot = zero

    def forward(self, arm_labels: torch.Tensor) -> torch.Tensor:
        """arm_labels: (B, 3)  0=left  1=right  -1=unknown → (B, 3, H)"""
        idx = arm_labels.clone()
        idx[idx < 0] = 2          # -1 → index-2 (zero embedding)
        return self.embedding(idx)


class SubGoalTokenizer(nn.Module):
    """Converts the CURRENT sub-goal (arm + color + substep) into a single
    hidden-size token that replaces the full T5 language condition.

    Core idea: instead of asking the model to infer arm-order from a compressed
    T5 sentence, we give it one explicit token encoding exactly what to do now:
        arm=left, color=red, step=0  →  single token → lang cross-attention K/V

    Indices:
        arm:   0=left  1=right  2=unknown
        color: 0=red   1=green  2=blue  3=unknown
        step:  0/1/2  (which of the 3 grasp steps)  3=unknown

    Unknown slots produce zero vectors (no-op in cross-attention).
    """
    ARM_MAP   = {"left": 0, "right": 1}          # unknown → 2
    COLOR_MAP = {"red block": 0, "green block": 1, "blue block": 2}  # unknown → 3
    UNKNOWN_ARM, UNKNOWN_COLOR, UNKNOWN_STEP = 2, 3, 3

    def __init__(self, hidden_size: int):
        super().__init__()
        self.arm_embed   = nn.Embedding(3, hidden_size)  # left/right/unknown
        self.color_embed = nn.Embedding(4, hidden_size)  # red/green/blue/unknown
        self.step_embed  = nn.Embedding(4, hidden_size)  # step 0/1/2/unknown
        self.norm = nn.LayerNorm(hidden_size)
        for emb in [self.arm_embed, self.color_embed, self.step_embed]:
            nn.init.normal_(emb.weight, std=0.02)
        # unknown indices → zero embedding (near-zero effect on cross-attn)
        nn.init.zeros_(self.arm_embed.weight[2])
        nn.init.zeros_(self.color_embed.weight[3])
        nn.init.zeros_(self.step_embed.weight[3])

    def forward(self, arm_idx: torch.Tensor, color_idx: torch.Tensor,
                step_idx: torch.Tensor) -> torch.Tensor:
        """
        arm_idx, color_idx, step_idx: each (B,) int64
        return: (B, 1, hidden_size)  — single goal token
        """
        token = (self.arm_embed(arm_idx) +
                 self.color_embed(color_idx) +
                 self.step_embed(step_idx))
        return self.norm(token).unsqueeze(1)


class RDTRunner(
        nn.Module, 
        CompatiblePyTorchModelHubMixin, 
        repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"
    ):
    def __init__(self, *, action_dim, pred_horizon, config, 
                 lang_token_dim, img_token_dim, state_token_dim, 
                 max_lang_cond_len, img_cond_len, lang_pos_embed_config=None, 
                 img_pos_embed_config=None, dtype=torch.bfloat16):
        super(RDTRunner, self).__init__()
        # Create diffusion model
        hidden_size = config['rdt']['hidden_size']
        self.model = RDT(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=config['rdt']['depth'],
            num_heads=config['rdt']['num_heads'],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
        )

        # Create adpators for various conditional inputs
        self.lang_adaptor = self.build_condition_adapter(
            config['lang_adaptor'], 
            in_features=lang_token_dim, 
            out_features=hidden_size
        )
        self.img_adaptor = self.build_condition_adapter(
            config['img_adaptor'], 
            in_features=img_token_dim, 
            out_features=hidden_size
        )
        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'], 
            in_features=state_token_dim * 2,    # state + state mask (indicator)
            out_features=hidden_size
        )

        # Optionally create Q-Former for language-guided visual refinement.
        # Image tokens (Q) attend to language tokens (K/V) after both adaptors,
        # so that visual features are grounded in task instruction before
        # being fed into the RDT Transformer.
        qformer_config = config.get('qformer', {})
        if qformer_config.get('enabled', False):
            self.qformer = LanguageGuidedQFormer(
                hidden_size=hidden_size,
                num_heads=config['rdt']['num_heads'],
                num_layers=qformer_config.get('num_layers', 2),
            )
        else:
            self.qformer = None

        # LangFiLM: language-conditioned FiLM on state/action tokens.
        # Forces every action prediction to be modulated by a pooled language
        # vector, preventing the model from ignoring language altogether.
        film_config = config.get('lang_film', {})
        if film_config.get('enabled', False):
            self.lang_film = LangFiLM(hidden_size)
        else:
            self.lang_film = None

        # Auxiliary arm-order prediction head.
        # Predicts which arm (left=0 / right=1) handles each of the 3 blocks
        # directly from the pooled language embedding, with a CE loss.
        # This creates a direct gradient path: language → arm ordering.
        aux_config = config.get('aux_arm_order', {})
        if aux_config.get('enabled', False):
            self.arm_order_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_size // 4, 6),  # 3 steps × 2 classes (left/right)
            )
            self.arm_order_loss_weight = aux_config.get('loss_weight', 0.1)
        else:
            self.arm_order_head = None
            self.arm_order_loss_weight = 0.0

        # Create the noise scheduler
        noise_scheduler_config = config['noise_scheduler']
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
        )

        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        self.prediction_type = noise_scheduler_config['prediction_type']

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        # ArmOrderPrefix: 3 explicit per-step tokens prepended to the language
        # cross-attention K/V sequence. Hidden-space tokens that DIRECTLY encode
        # arm assignment for each grasp step, bypassing T5 comprehension entirely.
        # At test time with unseen instructions: pass arm_labels parsed from text,
        # or all-(-1) for unknown → near-zero tokens (no-op).
        arm_prefix_config = config.get('arm_order_prefix', {})
        if arm_prefix_config.get('enabled', False):
            self.arm_order_prefix = ArmOrderPrefix(hidden_size)
        else:
            self.arm_order_prefix = None

        # SubGoalTokenizer: the core fix for instruction-following in low-data
        # regimes. Replaces (or prepends to) the T5 language embedding with
        # a SINGLE explicit token: (arm, color, substep) → hidden_size vector.
        # In "replace" mode (default), the model receives only this one token
        # as its language condition — exactly what the current step needs.
        subgoal_config = config.get('subgoal_tokenizer', {})
        if subgoal_config.get('enabled', False):
            self.subgoal_tokenizer = SubGoalTokenizer(hidden_size)
            self.subgoal_mode = subgoal_config.get('mode', 'replace')
        else:
            self.subgoal_tokenizer = None
            self.subgoal_mode = 'replace'

        qformer_params  = list(self.qformer.parameters()) if self.qformer is not None else []
        film_params     = list(self.lang_film.parameters()) if self.lang_film is not None else []
        aux_params      = list(self.arm_order_head.parameters()) if self.arm_order_head is not None else []
        prefix_params   = list(self.arm_order_prefix.parameters()) if self.arm_order_prefix is not None else []
        subgoal_params  = list(self.subgoal_tokenizer.parameters()) if self.subgoal_tokenizer is not None else []
        print("Diffusion params: %e" % sum(
            [p.numel() for p in self.model.parameters()] +
            [p.numel() for p in self.lang_adaptor.parameters()] +
            [p.numel() for p in self.img_adaptor.parameters()] +
            [p.numel() for p in self.state_adaptor.parameters()] +
            [p.numel() for p in qformer_params] +
            [p.numel() for p in film_params] +
            [p.numel() for p in aux_params] +
            [p.numel() for p in prefix_params] +
            [p.numel() for p in subgoal_params]))
        if self.qformer is not None:
            print("  of which QFormer params: %e" % sum(p.numel() for p in qformer_params))
        if self.lang_film is not None:
            print("  of which LangFiLM params: %e" % sum(p.numel() for p in film_params))
        if self.arm_order_head is not None:
            print("  of which ArmOrderHead params: %e" % sum(p.numel() for p in aux_params))
        if self.arm_order_prefix is not None:
            print("  of which ArmOrderPrefix params: %e" % sum(p.numel() for p in prefix_params))
        if self.subgoal_tokenizer is not None:
            print("  of which SubGoalTokenizer params: %e" % sum(p.numel() for p in subgoal_params))
            print(f"  SubGoal mode: {self.subgoal_mode}")
    
    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens,
                         lang_attn_mask=None, arm_labels=None, subgoal=None):
        '''
        lang_tokens: (B, lang_len, lang_token_dim)
        img_tokens:  (B, img_len, img_token_dim)
        state_tokens:(B, state_len, state_token_dim)
        lang_attn_mask: (B, lang_len) bool — valid lang tokens
        arm_labels:  (B, 3) int64  0=left 1=right -1=unknown  (for ArmOrderPrefix)
        subgoal:     dict with keys "arm", "color", "step" each (B,) int64
                     When SubGoalTokenizer is enabled:
                       mode="replace" → lang_cond = [goal_token]  (T5 discarded)
                       mode="prepend" → lang_cond = [goal_token] + T5 tokens

        return: (adapted_lang, adapted_img, adapted_state, lang_attn_mask)
                lang_attn_mask may be extended when tokens are prepended.
        '''
        adpated_lang = self.lang_adaptor(lang_tokens)
        adpated_img  = self.img_adaptor(img_tokens)
        adpated_state = self.state_adaptor(state_tokens)

        B      = lang_tokens.shape[0]
        device = lang_tokens.device

        # === SubGoalTokenizer: explicit (arm, color, step) → single token ===
        if self.subgoal_tokenizer is not None and subgoal is not None:
            goal_token = self.subgoal_tokenizer(
                subgoal["arm"].to(device),
                subgoal["color"].to(device),
                subgoal["step"].to(device),
            )  # (B, 1, H)

            if self.subgoal_mode == "replace":
                # Discard T5 tokens entirely — single goal token is the whole
                # language condition. This is the strongest form of instruction
                # injection: the model cannot ignore the arm/color/step signal.
                adpated_lang = goal_token
                lang_attn_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
            else:  # prepend — keep T5 context, goal token at front
                adpated_lang = torch.cat([goal_token, adpated_lang], dim=1)
                if lang_attn_mask is not None:
                    goal_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
                    lang_attn_mask = torch.cat([goal_mask, lang_attn_mask], dim=1)

        # ArmOrderPrefix: only active when SubGoalTokenizer is NOT replacing
        elif self.arm_order_prefix is not None and arm_labels is not None:
            prefix = self.arm_order_prefix(arm_labels)  # (B, 3, H)
            adpated_lang = torch.cat([prefix, adpated_lang], dim=1)
            if lang_attn_mask is not None:
                prefix_mask = arm_labels.ge(0).to(device)
                lang_attn_mask = torch.cat([prefix_mask, lang_attn_mask], dim=1)

        # Language-guided visual refinement via Q-Former
        if self.qformer is not None:
            adpated_img = self.qformer(adpated_img, adpated_lang, lang_attn_mask)

        # LangFiLM: modulate state/action tokens with pooled language vector.
        if self.lang_film is not None:
            if lang_attn_mask is not None:
                mask_f = lang_attn_mask.float().unsqueeze(-1)
                lang_pooled = (adpated_lang * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            else:
                lang_pooled = adpated_lang.mean(1)
            adpated_state = self.lang_film(adpated_state, lang_pooled)

        return adpated_lang, adpated_img, adpated_state, lang_attn_mask

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, 
                           state_traj, action_mask, ctrl_freqs):
        '''
        lang_cond: language conditional data, (batch_size, lang_len, hidden_size).
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_cond: image conditional data, (batch_size, img_len, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        action_mask: (batch_size, 1, action_dim), a 0-1 **float** tensor
            indicating the valid action dimensions.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(
            size=(state_traj.shape[0], self.pred_horizon, self.action_dim), 
            dtype=dtype, device=device)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)

        # Pre-compute pooled language vector for LangFiLM applied to action
        # tokens inside the loop — this fixes the train/inference mismatch
        # where FiLM was applied to the full state+action sequence during
        # training but only to the state token during inference.
        if self.lang_film is not None:
            if lang_attn_mask is not None:
                mask_f = lang_attn_mask.float().unsqueeze(-1)
                _lang_pooled = (lang_cond * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            else:
                _lang_pooled = lang_cond.mean(1)
        else:
            _lang_pooled = None
    
        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        
        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            # Apply LangFiLM to action tokens — consistent with training
            if _lang_pooled is not None:
                action_traj = self.lang_film(action_traj, _lang_pooled)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            
            # Predict the model output
            model_output = self.model(state_action_traj, ctrl_freqs,
                                    t.unsqueeze(-1).to(device),
                                    lang_cond, img_cond, lang_mask=lang_attn_mask)
            
            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)
        
        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action * action_mask

        return noisy_action
    
    # ========= Train  ============
    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, 
                     state_tokens, action_gt, action_mask, ctrl_freqs,
                     arm_labels=None, subgoal=None,
                    ) -> torch.Tensor:
        '''
        lang_tokens: (B, lang_len, lang_token_dim)
        lang_attn_mask: (B, lang_len) bool
        img_tokens: (B, img_len, img_token_dim)
        state_tokens: (B, 1, state_token_dim)
        action_gt: (B, horizon, state_token_dim)
        action_mask: (B, 1, state_token_dim) float 0/1
        ctrl_freqs: (B,)
        arm_labels: (B, 3) int64  0=left 1=right -1=unknown (for AuxCE + ArmOrderPrefix)
        subgoal: dict {"arm":(B,), "color":(B,), "step":(B,)} int64
                 When SubGoalTokenizer is enabled this replaces/prepends to T5 tokens.
        '''
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device  

        noise = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=device).long()
        noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)
        
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)

        lang_cond, img_cond, state_action_traj, lang_attn_mask = self.adapt_conditions(
            lang_tokens, img_tokens, state_action_traj, lang_attn_mask, arm_labels, subgoal)

        pred = self.model(state_action_traj, ctrl_freqs, 
                          timesteps, lang_cond, img_cond, 
                          lang_mask=lang_attn_mask)

        if self.prediction_type == 'epsilon':
            target = noise
        elif self.prediction_type == 'sample':
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {self.prediction_type}")

        loss = F.mse_loss(pred, target)

        # Auxiliary arm-order CE loss — works regardless of subgoal mode.
        # In replace mode the pooled lang_cond IS the goal token, so the CE
        # loss reinforces the arm embedding to reflect the true arm label.
        if self.arm_order_head is not None and arm_labels is not None:
            mask_f = lang_attn_mask.float().unsqueeze(-1)
            lang_pooled = (lang_cond * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            arm_logits = self.arm_order_head(lang_pooled).view(batch_size, 3, 2)
            valid = arm_labels >= 0
            if valid.any():
                aux_loss = F.cross_entropy(arm_logits[valid], arm_labels[valid].to(device))
                loss = loss + self.arm_order_loss_weight * aux_loss

        return loss
    
    # ========= Inference  ============
    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
                       action_mask, ctrl_freqs, arm_labels=None, subgoal=None):
        '''
        lang_tokens: (B, lang_len, lang_token_dim)
        lang_attn_mask: (B, lang_len) bool
        img_tokens: (B, img_len, img_token_dim)
        state_tokens: (B, 1, state_token_dim)
        action_mask: (B, 1, action_dim) float 0/1
        ctrl_freqs: (B,)
        arm_labels: (B, 3) int64  0=left 1=right -1=unknown  (for ArmOrderPrefix)
        subgoal: dict {"arm":(B,), "color":(B,), "step":(B,)} int64
                 Parse at inference time:
                   arm/color from scene_info, step from episode_progress * 3
        '''
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj, lang_attn_mask = self.adapt_conditions(
            lang_tokens, img_tokens, state_tokens, lang_attn_mask, arm_labels, subgoal)
        action_pred = self.conditional_sample(
            lang_cond, lang_attn_mask, img_cond, 
            state_traj, action_mask, ctrl_freqs,
        )
        return action_pred
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)

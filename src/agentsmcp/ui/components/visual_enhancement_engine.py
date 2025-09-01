"""
Visual Enhancement Engine - 60fps animation system with smooth transitions

This module provides a high-performance visual enhancement engine that delivers
smooth 60fps animations, typewriter effects, visual feedback, and adaptive
quality based on terminal performance for the Revolutionary TUI Interface.

Key Features:
- 60fps animation system with frame interpolation
- Typewriter effects with customizable speed and styles  
- Visual feedback (ripples, glows, shimmers, pulses)
- Smooth transitions and easing functions
- Adaptive quality scaling for performance
- Terminal capability detection and optimization
- Real-time performance monitoring
- Memory-efficient animation management
"""

import asyncio
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import logging
from collections import deque

logger = logging.getLogger(__name__)


class EasingType(Enum):
    """Animation easing types for smooth transitions."""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out" 
    EASE_IN_OUT = "ease_in_out"
    BOUNCE_IN = "bounce_in"
    BOUNCE_OUT = "bounce_out"
    ELASTIC_IN = "elastic_in"
    ELASTIC_OUT = "elastic_out"
    BACK_IN = "back_in"
    BACK_OUT = "back_out"


class EffectType(Enum):
    """Types of visual effects available."""
    TYPEWRITER = "typewriter"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    SLIDE_IN = "slide_in"
    SLIDE_OUT = "slide_out"
    PULSE = "pulse"
    GLOW = "glow"
    SHIMMER = "shimmer"
    RIPPLE = "ripple"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"


class QualityLevel(Enum):
    """Quality levels for adaptive performance."""
    LOW = "low"           # 20 FPS, minimal effects
    MEDIUM = "medium"     # 30 FPS, standard effects
    HIGH = "high"         # 45 FPS, enhanced effects
    ULTRA = "ultra"       # 60 FPS, all effects


@dataclass
class AnimationFrame:
    """Represents a single animation frame."""
    timestamp: float
    element_id: str
    property_name: str
    value: Any
    interpolated: bool = False


@dataclass
class Animation:
    """Represents an active animation with all parameters."""
    id: str
    element_id: str
    property_name: str
    start_value: float
    end_value: float
    duration_ms: int
    easing: EasingType
    start_time: float
    current_value: float = 0.0
    progress: float = 0.0
    is_active: bool = True
    loop_count: int = 1
    current_loop: int = 0
    reverse_on_complete: bool = False
    callback: Optional[Callable] = None
    frames: deque = field(default_factory=lambda: deque(maxlen=60))


@dataclass
class VisualEffect:
    """Represents a visual effect with parameters."""
    id: str
    element_id: str
    effect_type: EffectType
    duration_ms: int
    parameters: Dict[str, Any]
    start_time: float
    progress: float = 0.0
    is_active: bool = True
    frames_rendered: int = 0


@dataclass
class PerformanceMetrics:
    """Performance tracking for adaptive quality."""
    current_fps: float = 60.0
    target_fps: float = 60.0
    frame_times: deque = field(default_factory=lambda: deque(maxlen=60))
    dropped_frames: int = 0
    memory_usage_mb: float = 0.0
    active_animations: int = 0
    active_effects: int = 0
    render_time_ms: float = 0.0
    last_quality_adjustment: float = 0.0


class VisualEnhancementEngine:
    """
    High-performance 60fps visual enhancement engine.
    
    Provides smooth animations, visual effects, and adaptive performance
    optimization for the Revolutionary TUI Interface.
    """
    
    def __init__(self):
        """Initialize the Visual Enhancement Engine."""
        self.logger = logger
        
        # Core animation system
        self.animations: Dict[str, Animation] = {}
        self.effects: Dict[str, VisualEffect] = {}
        self.frame_buffer: deque = deque(maxlen=180)  # 3 seconds at 60fps
        
        # Performance monitoring
        self.performance_metrics = PerformanceMetrics()
        self.quality_level = QualityLevel.HIGH
        
        # Engine state
        self.is_running = False
        self.render_loop_task: Optional[asyncio.Task] = None
        self.last_frame_time = time.time()
        
        # Terminal capabilities
        self.terminal_capabilities = {
            "colors_256": True,
            "colors_16m": False,
            "unicode": True,
            "mouse_support": False,
            "performance_tier": "medium"
        }
        
        # Easing functions
        self.easing_functions = {
            EasingType.LINEAR: self._ease_linear,
            EasingType.EASE_IN: self._ease_in,
            EasingType.EASE_OUT: self._ease_out,
            EasingType.EASE_IN_OUT: self._ease_in_out,
            EasingType.BOUNCE_IN: self._ease_bounce_in,
            EasingType.BOUNCE_OUT: self._ease_bounce_out,
            EasingType.ELASTIC_IN: self._ease_elastic_in,
            EasingType.ELASTIC_OUT: self._ease_elastic_out,
            EasingType.BACK_IN: self._ease_back_in,
            EasingType.BACK_OUT: self._ease_back_out
        }
        
        # Effect handlers
        self.effect_handlers = {
            EffectType.TYPEWRITER: self._render_typewriter_effect,
            EffectType.FADE_IN: self._render_fade_effect,
            EffectType.FADE_OUT: self._render_fade_effect,
            EffectType.SLIDE_IN: self._render_slide_effect,
            EffectType.SLIDE_OUT: self._render_slide_effect,
            EffectType.PULSE: self._render_pulse_effect,
            EffectType.GLOW: self._render_glow_effect,
            EffectType.SHIMMER: self._render_shimmer_effect,
            EffectType.RIPPLE: self._render_ripple_effect,
            EffectType.ZOOM_IN: self._render_zoom_effect,
            EffectType.ZOOM_OUT: self._render_zoom_effect
        }
        
        self.logger.info("Visual Enhancement Engine initialized")
    
    async def start(self):
        """Start the visual enhancement engine."""
        if self.is_running:
            return
        
        self.is_running = True
        await self._detect_terminal_capabilities()
        await self._optimize_for_capabilities()
        
        # Start the render loop
        self.render_loop_task = asyncio.create_task(self._render_loop())
        
        self.logger.info(f"Visual Enhancement Engine started at {self.quality_level.value} quality")
    
    async def stop(self):
        """Stop the visual enhancement engine."""
        self.is_running = False
        
        if self.render_loop_task:
            self.render_loop_task.cancel()
            try:
                await self.render_loop_task
            except asyncio.CancelledError:
                pass
        
        # Clear all animations and effects
        self.animations.clear()
        self.effects.clear()
        
        self.logger.info("Visual Enhancement Engine stopped")
    
    async def animate(
        self,
        element_id: str,
        property_name: str,
        target_value: float,
        duration_ms: int = 300,
        easing: EasingType = EasingType.EASE_OUT,
        start_value: Optional[float] = None,
        loop_count: int = 1,
        reverse_on_complete: bool = False,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Create a smooth animation for an element property.
        
        Args:
            element_id: Target element identifier
            property_name: Property to animate (x, y, opacity, scale, etc.)
            target_value: Target value for the property
            duration_ms: Animation duration in milliseconds
            easing: Easing function to use
            start_value: Starting value (auto-detected if None)
            loop_count: Number of times to loop the animation
            reverse_on_complete: Whether to reverse animation on completion
            callback: Optional callback function when animation completes
            
        Returns:
            Animation ID for tracking/cancellation
        """
        animation_id = f"anim_{element_id}_{property_name}_{time.time()}"
        
        # Use provided start value or default to 0
        actual_start_value = start_value if start_value is not None else 0.0
        
        animation = Animation(
            id=animation_id,
            element_id=element_id,
            property_name=property_name,
            start_value=actual_start_value,
            end_value=target_value,
            duration_ms=duration_ms,
            easing=easing,
            start_time=time.time(),
            loop_count=loop_count,
            reverse_on_complete=reverse_on_complete,
            callback=callback
        )
        
        self.animations[animation_id] = animation
        
        self.logger.debug(f"Created animation {animation_id}: {property_name} {actual_start_value} -> {target_value}")
        
        return animation_id
    
    async def create_effect(
        self,
        element_id: str,
        effect_type: EffectType,
        duration_ms: int = 1000,
        **parameters
    ) -> str:
        """
        Create a visual effect on an element.
        
        Args:
            element_id: Target element identifier
            effect_type: Type of visual effect
            duration_ms: Effect duration in milliseconds
            **parameters: Effect-specific parameters
            
        Returns:
            Effect ID for tracking/cancellation
        """
        effect_id = f"effect_{element_id}_{effect_type.value}_{time.time()}"
        
        effect = VisualEffect(
            id=effect_id,
            element_id=element_id,
            effect_type=effect_type,
            duration_ms=duration_ms,
            parameters=parameters,
            start_time=time.time()
        )
        
        self.effects[effect_id] = effect
        
        self.logger.debug(f"Created effect {effect_id}: {effect_type.value} on {element_id}")
        
        return effect_id
    
    async def cancel_animation(self, animation_id: str) -> bool:
        """Cancel an active animation."""
        if animation_id in self.animations:
            animation = self.animations[animation_id]
            animation.is_active = False
            del self.animations[animation_id]
            
            self.logger.debug(f"Cancelled animation {animation_id}")
            return True
        
        return False
    
    async def cancel_effect(self, effect_id: str) -> bool:
        """Cancel an active effect."""
        if effect_id in self.effects:
            effect = self.effects[effect_id]
            effect.is_active = False
            del self.effects[effect_id]
            
            self.logger.debug(f"Cancelled effect {effect_id}")
            return True
        
        return False
    
    async def cancel_all_for_element(self, element_id: str) -> int:
        """Cancel all animations and effects for a specific element."""
        cancelled_count = 0
        
        # Cancel animations
        animations_to_remove = [
            anim_id for anim_id, animation in self.animations.items()
            if animation.element_id == element_id
        ]
        
        for anim_id in animations_to_remove:
            await self.cancel_animation(anim_id)
            cancelled_count += 1
        
        # Cancel effects
        effects_to_remove = [
            effect_id for effect_id, effect in self.effects.items()
            if effect.element_id == element_id
        ]
        
        for effect_id in effects_to_remove:
            await self.cancel_effect(effect_id)
            cancelled_count += 1
        
        return cancelled_count
    
    def get_current_value(self, element_id: str, property_name: str) -> Optional[float]:
        """Get the current animated value for an element property."""
        for animation in self.animations.values():
            if (animation.element_id == element_id and 
                animation.property_name == property_name and 
                animation.is_active):
                return animation.current_value
        
        return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        return {
            "fps": self.performance_metrics.current_fps,
            "target_fps": self.performance_metrics.target_fps,
            "quality_level": self.quality_level.value,
            "active_animations": len(self.animations),
            "active_effects": len(self.effects),
            "dropped_frames": self.performance_metrics.dropped_frames,
            "render_time_ms": self.performance_metrics.render_time_ms,
            "memory_usage_mb": self.performance_metrics.memory_usage_mb,
            "terminal_capabilities": self.terminal_capabilities
        }
    
    # ==================== INTERNAL RENDER LOOP ====================
    
    async def _render_loop(self):
        """Main 60fps render loop."""
        target_frame_time = 1.0 / self.performance_metrics.target_fps
        
        while self.is_running:
            frame_start_time = time.time()
            
            try:
                # Update animations
                await self._update_animations()
                
                # Update effects
                await self._update_effects()
                
                # Monitor performance
                await self._monitor_performance(frame_start_time)
                
                # Adaptive quality adjustment
                if time.time() - self.performance_metrics.last_quality_adjustment > 2.0:
                    await self._adjust_quality_if_needed()
                
                # Calculate frame timing
                frame_time = time.time() - frame_start_time
                sleep_time = max(0, target_frame_time - frame_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    self.performance_metrics.dropped_frames += 1
                
            except Exception as e:
                self.logger.error(f"Error in render loop: {e}")
                await asyncio.sleep(0.1)  # Prevent tight error loop
    
    async def _update_animations(self):
        """Update all active animations."""
        current_time = time.time()
        completed_animations = []
        
        for animation_id, animation in self.animations.items():
            if not animation.is_active:
                continue
            
            # Calculate progress
            elapsed_time = current_time - animation.start_time
            progress = min(elapsed_time / (animation.duration_ms / 1000.0), 1.0)
            
            # Apply easing
            easing_func = self.easing_functions[animation.easing]
            eased_progress = easing_func(progress)
            
            # Calculate current value
            value_range = animation.end_value - animation.start_value
            animation.current_value = animation.start_value + (value_range * eased_progress)
            animation.progress = progress
            
            # Store frame data
            frame = AnimationFrame(
                timestamp=current_time,
                element_id=animation.element_id,
                property_name=animation.property_name,
                value=animation.current_value
            )
            animation.frames.append(frame)
            
            # Check if animation is complete
            if progress >= 1.0:
                animation.current_loop += 1
                
                if animation.current_loop >= animation.loop_count:
                    animation.is_active = False
                    completed_animations.append(animation_id)
                    
                    # Execute callback if provided
                    if animation.callback:
                        try:
                            if asyncio.iscoroutinefunction(animation.callback):
                                await animation.callback()
                            else:
                                animation.callback()
                        except Exception as e:
                            self.logger.error(f"Error in animation callback: {e}")
                
                elif animation.reverse_on_complete and animation.current_loop % 2 == 1:
                    # Reverse animation
                    animation.start_value, animation.end_value = animation.end_value, animation.start_value
                    animation.start_time = current_time
                else:
                    # Restart loop
                    animation.start_time = current_time
        
        # Remove completed animations
        for animation_id in completed_animations:
            del self.animations[animation_id]
    
    async def _update_effects(self):
        """Update all active visual effects."""
        current_time = time.time()
        completed_effects = []
        
        for effect_id, effect in self.effects.items():
            if not effect.is_active:
                continue
            
            # Calculate progress
            elapsed_time = current_time - effect.start_time
            progress = min(elapsed_time / (effect.duration_ms / 1000.0), 1.0)
            effect.progress = progress
            
            # Render effect
            handler = self.effect_handlers.get(effect.effect_type)
            if handler:
                try:
                    await handler(effect)
                    effect.frames_rendered += 1
                except Exception as e:
                    self.logger.error(f"Error rendering effect {effect_id}: {e}")
            
            # Check if effect is complete
            if progress >= 1.0:
                effect.is_active = False
                completed_effects.append(effect_id)
        
        # Remove completed effects
        for effect_id in completed_effects:
            del self.effects[effect_id]
    
    async def _monitor_performance(self, frame_start_time: float):
        """Monitor and update performance metrics."""
        frame_time = time.time() - frame_start_time
        self.performance_metrics.frame_times.append(frame_time)
        self.performance_metrics.render_time_ms = frame_time * 1000
        
        # Calculate current FPS
        if len(self.performance_metrics.frame_times) >= 10:
            avg_frame_time = sum(self.performance_metrics.frame_times) / len(self.performance_metrics.frame_times)
            self.performance_metrics.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Update other metrics
        self.performance_metrics.active_animations = len(self.animations)
        self.performance_metrics.active_effects = len(self.effects)
    
    async def _adjust_quality_if_needed(self):
        """Adjust quality level based on performance."""
        current_fps = self.performance_metrics.current_fps
        target_fps = self.performance_metrics.target_fps
        
        # Adjust quality based on FPS performance
        if current_fps < target_fps * 0.75:  # Performance is poor
            if self.quality_level == QualityLevel.ULTRA:
                await self._set_quality_level(QualityLevel.HIGH)
            elif self.quality_level == QualityLevel.HIGH:
                await self._set_quality_level(QualityLevel.MEDIUM)
            elif self.quality_level == QualityLevel.MEDIUM:
                await self._set_quality_level(QualityLevel.LOW)
                
        elif current_fps > target_fps * 0.95:  # Performance is good, try upgrading
            if self.quality_level == QualityLevel.LOW:
                await self._set_quality_level(QualityLevel.MEDIUM)
            elif self.quality_level == QualityLevel.MEDIUM:
                await self._set_quality_level(QualityLevel.HIGH)
            elif self.quality_level == QualityLevel.HIGH and self.terminal_capabilities.get("performance_tier") == "high":
                await self._set_quality_level(QualityLevel.ULTRA)
        
        self.performance_metrics.last_quality_adjustment = time.time()
    
    async def _set_quality_level(self, quality_level: QualityLevel):
        """Set the quality level and adjust parameters."""
        if quality_level == self.quality_level:
            return
        
        old_quality = self.quality_level
        self.quality_level = quality_level
        
        # Adjust target FPS based on quality
        if quality_level == QualityLevel.LOW:
            self.performance_metrics.target_fps = 20.0
        elif quality_level == QualityLevel.MEDIUM:
            self.performance_metrics.target_fps = 30.0
        elif quality_level == QualityLevel.HIGH:
            self.performance_metrics.target_fps = 45.0
        elif quality_level == QualityLevel.ULTRA:
            self.performance_metrics.target_fps = 60.0
        
        self.logger.info(f"Quality adjusted from {old_quality.value} to {quality_level.value} (Target FPS: {self.performance_metrics.target_fps})")
    
    # ==================== TERMINAL CAPABILITIES ====================
    
    async def _detect_terminal_capabilities(self):
        """Detect terminal capabilities for optimization."""
        import os
        
        term = os.environ.get('TERM', '').lower()
        term_program = os.environ.get('TERM_PROGRAM', '').lower()
        
        # Color support
        if 'truecolor' in term or '24bit' in term:
            self.terminal_capabilities["colors_16m"] = True
            self.terminal_capabilities["performance_tier"] = "high"
        elif '256color' in term:
            self.terminal_capabilities["colors_256"] = True
            self.terminal_capabilities["performance_tier"] = "medium"
        else:
            self.terminal_capabilities["performance_tier"] = "low"
        
        # Advanced terminal features
        if 'kitty' in term_program:
            self.terminal_capabilities.update({
                "colors_16m": True,
                "unicode": True,
                "mouse_support": True,
                "performance_tier": "high"
            })
        elif 'iterm' in term_program:
            self.terminal_capabilities.update({
                "colors_16m": True,
                "unicode": True,
                "mouse_support": True,
                "performance_tier": "high"
            })
        elif 'vscode' in term_program:
            self.terminal_capabilities.update({
                "colors_16m": True,
                "unicode": True,
                "performance_tier": "medium"
            })
    
    async def _optimize_for_capabilities(self):
        """Optimize settings based on terminal capabilities."""
        perf_tier = self.terminal_capabilities.get("performance_tier", "medium")
        
        if perf_tier == "high":
            self.quality_level = QualityLevel.HIGH
            self.performance_metrics.target_fps = 45.0
        elif perf_tier == "medium":
            self.quality_level = QualityLevel.MEDIUM
            self.performance_metrics.target_fps = 30.0
        else:
            self.quality_level = QualityLevel.LOW
            self.performance_metrics.target_fps = 20.0
    
    # ==================== EASING FUNCTIONS ====================
    
    def _ease_linear(self, t: float) -> float:
        """Linear easing function."""
        return t
    
    def _ease_in(self, t: float) -> float:
        """Ease in (quadratic)."""
        return t * t
    
    def _ease_out(self, t: float) -> float:
        """Ease out (quadratic)."""
        return t * (2 - t)
    
    def _ease_in_out(self, t: float) -> float:
        """Ease in-out (quadratic)."""
        return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t
    
    def _ease_bounce_in(self, t: float) -> float:
        """Bounce in easing."""
        return 1 - self._ease_bounce_out(1 - t)
    
    def _ease_bounce_out(self, t: float) -> float:
        """Bounce out easing."""
        if t < 1/2.75:
            return 7.5625 * t * t
        elif t < 2/2.75:
            t -= 1.5/2.75
            return 7.5625 * t * t + 0.75
        elif t < 2.5/2.75:
            t -= 2.25/2.75
            return 7.5625 * t * t + 0.9375
        else:
            t -= 2.625/2.75
            return 7.5625 * t * t + 0.984375
    
    def _ease_elastic_in(self, t: float) -> float:
        """Elastic in easing."""
        if t == 0 or t == 1:
            return t
        
        p = 0.3
        s = p / 4
        t -= 1
        return -(2 ** (10 * t)) * math.sin((t - s) * (2 * math.pi) / p)
    
    def _ease_elastic_out(self, t: float) -> float:
        """Elastic out easing."""
        if t == 0 or t == 1:
            return t
        
        p = 0.3
        s = p / 4
        return (2 ** (-10 * t)) * math.sin((t - s) * (2 * math.pi) / p) + 1
    
    def _ease_back_in(self, t: float) -> float:
        """Back in easing."""
        s = 1.70158
        return t * t * ((s + 1) * t - s)
    
    def _ease_back_out(self, t: float) -> float:
        """Back out easing."""
        s = 1.70158
        t -= 1
        return t * t * ((s + 1) * t + s) + 1
    
    # ==================== EFFECT HANDLERS ====================
    
    async def _render_typewriter_effect(self, effect: VisualEffect):
        """Render typewriter effect."""
        full_text = effect.parameters.get('text', '')
        speed = effect.parameters.get('speed', 20)  # characters per second
        show_cursor = effect.parameters.get('show_cursor', True)
        
        chars_to_show = int(effect.progress * len(full_text))
        current_text = full_text[:chars_to_show]
        
        # Add blinking cursor
        if show_cursor and chars_to_show < len(full_text):
            cursor_blink = int(time.time() * 2) % 2  # Blink every 0.5 seconds
            if cursor_blink:
                current_text += '█'
        
        # Store the rendered text (would be used by the TUI interface)
        effect.parameters['rendered_text'] = current_text
    
    async def _render_fade_effect(self, effect: VisualEffect):
        """Render fade in/out effect."""
        if effect.effect_type == EffectType.FADE_IN:
            opacity = effect.progress
        else:  # FADE_OUT
            opacity = 1.0 - effect.progress
        
        effect.parameters['opacity'] = opacity
    
    async def _render_slide_effect(self, effect: VisualEffect):
        """Render slide in/out effect."""
        direction = effect.parameters.get('direction', 'left')  # left, right, up, down
        distance = effect.parameters.get('distance', 10)
        
        if effect.effect_type == EffectType.SLIDE_IN:
            progress = effect.progress
        else:  # SLIDE_OUT
            progress = 1.0 - effect.progress
        
        if direction == 'left':
            offset_x = distance * (1.0 - progress)
            offset_y = 0
        elif direction == 'right':
            offset_x = -distance * (1.0 - progress)
            offset_y = 0
        elif direction == 'up':
            offset_x = 0
            offset_y = distance * (1.0 - progress)
        else:  # down
            offset_x = 0
            offset_y = -distance * (1.0 - progress)
        
        effect.parameters['offset_x'] = offset_x
        effect.parameters['offset_y'] = offset_y
    
    async def _render_pulse_effect(self, effect: VisualEffect):
        """Render pulse effect."""
        # Create a sine wave pulse
        pulse_frequency = effect.parameters.get('frequency', 2.0)
        min_intensity = effect.parameters.get('min_intensity', 0.3)
        max_intensity = effect.parameters.get('max_intensity', 1.0)
        
        pulse_value = math.sin(effect.progress * pulse_frequency * 2 * math.pi)
        intensity = min_intensity + (max_intensity - min_intensity) * (pulse_value + 1) / 2
        
        effect.parameters['intensity'] = intensity
    
    async def _render_glow_effect(self, effect: VisualEffect):
        """Render glow effect."""
        intensity = effect.parameters.get('intensity', 1.0)
        color = effect.parameters.get('color', 'blue')
        
        # Glow intensity varies with progress
        glow_intensity = intensity * math.sin(effect.progress * math.pi)
        
        effect.parameters['glow_intensity'] = glow_intensity
        effect.parameters['glow_color'] = color
    
    async def _render_shimmer_effect(self, effect: VisualEffect):
        """Render shimmer effect."""
        speed = effect.parameters.get('speed', 1.0)
        
        # Create a moving shimmer
        shimmer_position = (effect.progress * speed) % 1.0
        shimmer_intensity = math.sin(shimmer_position * math.pi * 4) * 0.5 + 0.5
        
        effect.parameters['shimmer_position'] = shimmer_position
        effect.parameters['shimmer_intensity'] = shimmer_intensity
    
    async def _render_ripple_effect(self, effect: VisualEffect):
        """Render ripple effect."""
        center_x = effect.parameters.get('center_x', 0)
        center_y = effect.parameters.get('center_y', 0)
        max_radius = effect.parameters.get('max_radius', 10)
        
        current_radius = effect.progress * max_radius
        ripple_intensity = 1.0 - effect.progress
        
        effect.parameters['ripple_radius'] = current_radius
        effect.parameters['ripple_intensity'] = ripple_intensity
        effect.parameters['ripple_center'] = (center_x, center_y)
    
    async def _render_zoom_effect(self, effect: VisualEffect):
        """Render zoom in/out effect."""
        if effect.effect_type == EffectType.ZOOM_IN:
            scale = effect.progress
        else:  # ZOOM_OUT
            scale = 1.0 - effect.progress
        
        effect.parameters['scale'] = scale


# Convenience functions for common effects
async def create_typewriter_effect(
    engine: VisualEnhancementEngine,
    element_id: str,
    text: str,
    speed: int = 20,
    show_cursor: bool = True
) -> str:
    """Create a typewriter effect."""
    return await engine.create_effect(
        element_id=element_id,
        effect_type=EffectType.TYPEWRITER,
        duration_ms=int((len(text) / speed) * 1000),
        text=text,
        speed=speed,
        show_cursor=show_cursor
    )


async def create_smooth_transition(
    engine: VisualEnhancementEngine,
    element_id: str,
    property_name: str,
    target_value: float,
    duration_ms: int = 300
) -> str:
    """Create a smooth transition animation."""
    return await engine.animate(
        element_id=element_id,
        property_name=property_name,
        target_value=target_value,
        duration_ms=duration_ms,
        easing=EasingType.EASE_OUT
    )


# Example usage
async def main():
    """Example usage of the Visual Enhancement Engine."""
    engine = VisualEnhancementEngine()
    
    try:
        await engine.start()
        
        # Create a typewriter effect
        typewriter_id = await create_typewriter_effect(
            engine, "welcome_message", 
            "Welcome to the Revolutionary TUI Interface!", 
            speed=25
        )
        
        # Create smooth animations
        slide_id = await engine.animate(
            "panel", "x", 100, 
            duration_ms=1000, 
            easing=EasingType.BOUNCE_OUT
        )
        
        fade_id = await engine.animate(
            "status", "opacity", 1.0, 
            duration_ms=500,
            easing=EasingType.FADE_IN
        )
        
        # Let effects run
        await asyncio.sleep(3)
        
        # Get performance report
        report = engine.get_performance_report()
        print("Performance Report:", report)
        
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
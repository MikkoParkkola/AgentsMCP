"""Modal coordinator for cross-modal interface transitions.

This module provides the main coordination layer for seamless switching
between CLI, TUI, WebUI, and API interfaces while preserving session state.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from .state_synchronizer import StateSynchronizer
from .capability_manager import CapabilityManager
from ..models.coordination_models import (
    InterfaceMode,
    ModalSwitchRequest,
    TransitionResult,
    StateSync,
    CapabilityQuery,
    SharedState,
    SyncStatus,
    ConflictResolution,
    StateChange,
    SessionContext,
    Feature,
    CapabilityInfo,
    ModeNotSupportedError,
    StateLossError,
    SyncFailedError,
    CapabilityMismatchError
)

logger = logging.getLogger(__name__)


class ModalCoordinator:
    """Main coordinator for cross-modal interface management."""
    
    def __init__(
        self, 
        storage_path: str = "~/.agentsmcp/coordination",
        encryption_key: Optional[str] = None
    ):
        """Initialize modal coordinator."""
        self.state_synchronizer = StateSynchronizer(
            storage_path=f"{storage_path}/state",
            encryption_key=encryption_key
        )
        self.capability_manager = CapabilityManager()
        
        # Active interface sessions
        self._active_interfaces: Dict[str, InterfaceMode] = {}
        
        # Transition handlers for each interface mode
        self._transition_handlers: Dict[InterfaceMode, Any] = {}
        
        # Performance metrics
        self._transition_metrics: Dict[str, List[float]] = {
            'transition_duration_ms': [],
            'state_sync_duration_ms': [],
            'context_preservation_rate': []
        }
        
        logger.info("Modal coordinator initialized")
    
    async def initialize(self) -> None:
        """Initialize coordinator and detect available interfaces."""
        logger.info("Initializing modal coordinator")
        
        # Detect environment capabilities
        env_capabilities = await self.capability_manager.detect_environment_capabilities()
        logger.debug(f"Environment capabilities: {env_capabilities}")
        
        # Initialize interface handlers based on available capabilities
        await self._initialize_interface_handlers(env_capabilities)
        
        logger.info("Modal coordinator initialization complete")
    
    async def _initialize_interface_handlers(self, env_capabilities: Dict[str, Any]) -> None:
        """Initialize handlers for available interfaces."""
        # CLI is always available
        self._transition_handlers[InterfaceMode.CLI] = await self._create_cli_handler()
        
        # TUI requires TTY
        if env_capabilities.get('has_tty', False):
            self._transition_handlers[InterfaceMode.TUI] = await self._create_tui_handler()
        
        # WebUI requires network
        if env_capabilities.get('network_available', False):
            self._transition_handlers[InterfaceMode.WEB_UI] = await self._create_webui_handler()
        
        # API is available if we can bind to network
        if env_capabilities.get('network_available', False):
            self._transition_handlers[InterfaceMode.API] = await self._create_api_handler()
        
        logger.info(f"Initialized handlers for: {list(self._transition_handlers.keys())}")
    
    async def _create_cli_handler(self) -> 'CLITransitionHandler':
        """Create CLI transition handler."""
        return CLITransitionHandler()
    
    async def _create_tui_handler(self) -> 'TUITransitionHandler':
        """Create TUI transition handler."""
        return TUITransitionHandler()
    
    async def _create_webui_handler(self) -> 'WebUITransitionHandler':
        """Create WebUI transition handler."""
        return WebUITransitionHandler()
    
    async def _create_api_handler(self) -> 'APITransitionHandler':
        """Create API transition handler."""
        return APITransitionHandler()
    
    async def switch_mode(self, request: ModalSwitchRequest) -> TransitionResult:
        """Switch interface modes with state preservation."""
        start_time = datetime.now(timezone.utc)
        logger.info(f"Switching from {request.from_mode} to {request.to_mode} for session {request.session_id}")
        
        try:
            # Validate transition is possible
            await self._validate_transition(request)
            
            # Get current state
            current_state = await self.state_synchronizer.get_session(request.session_id)
            if not current_state:
                raise StateLossError(f"Session {request.session_id} not found")
            
            # Extract context from source interface
            preserved_context = {}
            lost_context = []
            warnings = []
            
            if request.preserve_state:
                preserved_context, lost_context = await self._extract_interface_context(
                    request.from_mode, 
                    current_state
                )
            
            # Prepare target interface
            await self._prepare_target_interface(request.to_mode, current_state, preserved_context)
            
            # Update session context
            current_state.active_context.current_interface = request.to_mode
            current_state.active_context.last_activity = datetime.now(timezone.utc)
            
            # Add interface state if transferring context
            if request.transfer_context and preserved_context:
                current_state.interface_states[request.to_mode] = preserved_context
            
            # Synchronize state
            state_changes = [
                StateChange(
                    session_id=request.session_id,
                    interface_mode=request.to_mode,
                    key_path="active_context.current_interface",
                    old_value=request.from_mode.value,
                    new_value=request.to_mode.value,
                    user_id=current_state.active_context.user_id
                )
            ]
            
            sync_request = StateSync(
                session_id=request.session_id,
                interface_mode=request.to_mode,
                current_version=current_state.version,
                changes=state_changes,
                conflict_resolution=ConflictResolution.LAST_WRITE_WINS
            )
            
            sync_result = await self.state_synchronizer.synchronize_state(sync_request)
            if not sync_result.success:
                raise SyncFailedError(f"State sync failed: {sync_result.errors}")
            
            # Activate target interface
            await self._activate_interface(request.to_mode, current_state)
            
            # Update active interface tracking
            self._active_interfaces[request.session_id] = request.to_mode
            
            # Record metrics
            end_time = datetime.now(timezone.utc)
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            self._record_transition_metrics(duration_ms, len(preserved_context), len(lost_context))
            
            # Add warnings for lost context
            if lost_context:
                warnings.extend([f"Lost context: {item}" for item in lost_context])
            
            logger.info(f"Successfully switched to {request.to_mode} for session {request.session_id}")
            
            return TransitionResult(
                request_id=request.request_id,
                success=True,
                from_mode=request.from_mode,
                to_mode=request.to_mode,
                preserved_context=preserved_context,
                lost_context=lost_context,
                warnings=warnings,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"Mode switch failed: {e}")
            duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            
            return TransitionResult(
                request_id=request.request_id,
                success=False,
                from_mode=request.from_mode,
                to_mode=request.to_mode,
                errors=[str(e)],
                duration_ms=duration_ms
            )
    
    async def _validate_transition(self, request: ModalSwitchRequest) -> None:
        """Validate that transition is possible."""
        # Check if target mode is supported
        if request.to_mode not in self._transition_handlers:
            raise ModeNotSupportedError(f"Interface mode {request.to_mode} not available")
        
        # Check capabilities for target mode
        try:
            capability_query = CapabilityQuery(
                interface=request.to_mode,
                check_permissions=True,
                user_id=request.requested_by
            )
            await self.capability_manager.query_capabilities(capability_query)
        except CapabilityMismatchError as e:
            raise ModeNotSupportedError(f"Target mode capabilities not available: {e}")
    
    async def _extract_interface_context(
        self, 
        interface_mode: InterfaceMode,
        state: SharedState
    ) -> tuple[Dict[str, Any], List[str]]:
        """Extract context from source interface."""
        preserved_context = {}
        lost_context = []
        
        handler = self._transition_handlers.get(interface_mode)
        if not handler:
            lost_context.append(f"No handler for {interface_mode}")
            return preserved_context, lost_context
        
        try:
            # Get interface-specific state
            interface_state = state.interface_states.get(interface_mode, {})
            
            # Common context that can be preserved across interfaces
            transferable_keys = [
                'window_size', 'theme', 'user_preferences', 'command_history',
                'active_commands', 'current_directory', 'environment_vars'
            ]
            
            for key in transferable_keys:
                if key in interface_state:
                    preserved_context[key] = interface_state[key]
            
            # Interface-specific context extraction
            if hasattr(handler, 'extract_context'):
                interface_context = await handler.extract_context(state)
                preserved_context.update(interface_context)
            
        except Exception as e:
            logger.warning(f"Failed to extract context from {interface_mode}: {e}")
            lost_context.append(f"Context extraction failed: {e}")
        
        return preserved_context, lost_context
    
    async def _prepare_target_interface(
        self, 
        interface_mode: InterfaceMode,
        state: SharedState,
        context: Dict[str, Any]
    ) -> None:
        """Prepare target interface for activation."""
        handler = self._transition_handlers.get(interface_mode)
        if not handler:
            return
        
        try:
            if hasattr(handler, 'prepare_interface'):
                await handler.prepare_interface(state, context)
        except Exception as e:
            logger.warning(f"Failed to prepare {interface_mode}: {e}")
    
    async def _activate_interface(
        self, 
        interface_mode: InterfaceMode,
        state: SharedState
    ) -> None:
        """Activate target interface."""
        handler = self._transition_handlers.get(interface_mode)
        if not handler:
            return
        
        try:
            if hasattr(handler, 'activate_interface'):
                await handler.activate_interface(state)
        except Exception as e:
            logger.error(f"Failed to activate {interface_mode}: {e}")
            raise
    
    def _record_transition_metrics(
        self, 
        duration_ms: int, 
        preserved_count: int, 
        lost_count: int
    ) -> None:
        """Record transition performance metrics."""
        self._transition_metrics['transition_duration_ms'].append(duration_ms)
        
        if preserved_count + lost_count > 0:
            preservation_rate = preserved_count / (preserved_count + lost_count)
            self._transition_metrics['context_preservation_rate'].append(preservation_rate)
        
        # Keep only last 100 measurements
        for metric_list in self._transition_metrics.values():
            if len(metric_list) > 100:
                metric_list.pop(0)
    
    async def get_synchronized_state(self, session_id: str) -> Optional[SharedState]:
        """Get current synchronized state for session."""
        return await self.state_synchronizer.get_session(session_id)
    
    async def sync_state(self, sync_request: StateSync) -> Any:
        """Synchronize state across interfaces."""
        return await self.state_synchronizer.synchronize_state(sync_request)
    
    async def query_mode_capabilities(self, query: CapabilityQuery) -> CapabilityInfo:
        """Query capabilities for interface mode."""
        return await self.capability_manager.query_capabilities(query)
    
    async def get_all_mode_capabilities(self) -> Dict[InterfaceMode, List[Feature]]:
        """Get capability matrix for all supported interface modes."""
        return await self.capability_manager.get_mode_capabilities()
    
    async def create_session(
        self, 
        user_id: str, 
        initial_interface: InterfaceMode,
        session_id: Optional[str] = None
    ) -> SharedState:
        """Create new coordinated session."""
        if not session_id:
            from uuid import uuid4
            session_id = str(uuid4())
        
        # Validate initial interface is available
        if initial_interface not in self._transition_handlers:
            raise ModeNotSupportedError(f"Interface mode {initial_interface} not available")
        
        # Create session
        state = await self.state_synchronizer.create_session(
            session_id=session_id,
            user_id=user_id,
            interface_mode=initial_interface
        )
        
        # Track active interface
        self._active_interfaces[session_id] = initial_interface
        
        return state
    
    async def recover_session(self, session_id: str, user_id: str) -> Optional[SharedState]:
        """Recover session from storage."""
        state = await self.state_synchronizer.recover_session(session_id, user_id)
        if state:
            self._active_interfaces[session_id] = state.active_context.current_interface
        return state
    
    async def cleanup_session(self, session_id: str, preserve_state: bool = True) -> None:
        """Clean up session resources."""
        await self.state_synchronizer.cleanup_session(session_id, preserve_state)
        self._active_interfaces.pop(session_id, None)
    
    def get_current_interface(self, session_id: str) -> Optional[InterfaceMode]:
        """Get current interface for session."""
        return self._active_interfaces.get(session_id)
    
    def get_supported_interfaces(self) -> List[InterfaceMode]:
        """Get list of supported interface modes."""
        return list(self._transition_handlers.keys())
    
    def get_transition_metrics(self) -> Dict[str, Any]:
        """Get transition performance metrics."""
        metrics = {}
        for metric_name, values in self._transition_metrics.items():
            if values:
                metrics[metric_name] = {
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
            else:
                metrics[metric_name] = {'average': 0, 'min': 0, 'max': 0, 'count': 0}
        
        return metrics
    
    async def graceful_degradation(
        self, 
        session_id: str, 
        target_mode: InterfaceMode,
        required_features: List[str]
    ) -> List[str]:
        """Find alternative features when target mode is unavailable."""
        return await self.capability_manager.graceful_degradation(
            required_features, 
            target_mode
        )
    
    async def shutdown(self) -> None:
        """Shutdown coordinator and cleanup resources."""
        logger.info("Shutting down modal coordinator")
        
        # Cleanup all active sessions
        for session_id in list(self._active_interfaces.keys()):
            await self.cleanup_session(session_id, preserve_state=True)
        
        # Shutdown synchronizer
        await self.state_synchronizer.shutdown()
        
        # Clear handlers
        self._transition_handlers.clear()
        
        logger.info("Modal coordinator shutdown complete")


# Interface-specific transition handlers
class BaseTransitionHandler:
    """Base class for interface transition handlers."""
    
    async def extract_context(self, state: SharedState) -> Dict[str, Any]:
        """Extract interface-specific context."""
        return {}
    
    async def prepare_interface(self, state: SharedState, context: Dict[str, Any]) -> None:
        """Prepare interface for activation."""
        pass
    
    async def activate_interface(self, state: SharedState) -> None:
        """Activate interface."""
        pass


class CLITransitionHandler(BaseTransitionHandler):
    """Handler for CLI interface transitions."""
    
    async def extract_context(self, state: SharedState) -> Dict[str, Any]:
        """Extract CLI-specific context."""
        return {
            'exit_code': 0,
            'output_format': 'text',
            'verbosity_level': state.user_preferences.get('verbose_output', False)
        }
    
    async def prepare_interface(self, state: SharedState, context: Dict[str, Any]) -> None:
        """Prepare CLI interface."""
        # Set CLI-specific preferences
        if 'verbosity_level' in context:
            state.user_preferences['verbose_output'] = context['verbosity_level']


class TUITransitionHandler(BaseTransitionHandler):
    """Handler for TUI interface transitions."""
    
    async def extract_context(self, state: SharedState) -> Dict[str, Any]:
        """Extract TUI-specific context."""
        interface_state = state.interface_states.get(InterfaceMode.TUI, {})
        
        return {
            'window_size': interface_state.get('window_size', (80, 24)),
            'cursor_position': interface_state.get('cursor_position', (0, 0)),
            'active_panels': interface_state.get('active_panels', ['main']),
            'theme': interface_state.get('theme', 'default')
        }
    
    async def prepare_interface(self, state: SharedState, context: Dict[str, Any]) -> None:
        """Prepare TUI interface."""
        # Initialize TUI state
        tui_state = {
            'window_size': context.get('window_size', (80, 24)),
            'theme': context.get('theme', 'default'),
            'active_panels': context.get('active_panels', ['main'])
        }
        state.interface_states[InterfaceMode.TUI] = tui_state


class WebUITransitionHandler(BaseTransitionHandler):
    """Handler for WebUI interface transitions."""
    
    async def extract_context(self, state: SharedState) -> Dict[str, Any]:
        """Extract WebUI-specific context."""
        interface_state = state.interface_states.get(InterfaceMode.WEB_UI, {})
        
        return {
            'viewport_size': interface_state.get('viewport_size', (1920, 1080)),
            'active_tabs': interface_state.get('active_tabs', []),
            'sidebar_state': interface_state.get('sidebar_state', 'collapsed'),
            'theme': interface_state.get('theme', 'light')
        }
    
    async def prepare_interface(self, state: SharedState, context: Dict[str, Any]) -> None:
        """Prepare WebUI interface."""
        # Initialize web server if needed
        webui_state = {
            'server_port': 8080,
            'theme': context.get('theme', 'light'),
            'sidebar_state': context.get('sidebar_state', 'collapsed')
        }
        state.interface_states[InterfaceMode.WEB_UI] = webui_state


class APITransitionHandler(BaseTransitionHandler):
    """Handler for API interface transitions."""
    
    async def extract_context(self, state: SharedState) -> Dict[str, Any]:
        """Extract API-specific context."""
        interface_state = state.interface_states.get(InterfaceMode.API, {})
        
        return {
            'api_version': interface_state.get('api_version', 'v1'),
            'response_format': interface_state.get('response_format', 'json'),
            'rate_limits': interface_state.get('rate_limits', {}),
            'authentication': interface_state.get('authentication', {})
        }
    
    async def prepare_interface(self, state: SharedState, context: Dict[str, Any]) -> None:
        """Prepare API interface."""
        # Initialize API server configuration
        api_state = {
            'api_version': context.get('api_version', 'v1'),
            'response_format': context.get('response_format', 'json'),
            'server_port': 8090
        }
        state.interface_states[InterfaceMode.API] = api_state
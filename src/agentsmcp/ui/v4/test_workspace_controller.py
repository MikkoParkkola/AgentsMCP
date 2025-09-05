"""
Test suite for the V4 Workspace Controller

Comprehensive tests covering:
- Persistent event loop behavior
- Non-blocking input handling
- Agent orchestration functionality
- UI rendering and layout
- Keyboard shortcuts and controls
- Edge cases and error handling
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from collections import deque

from .workspace_controller import (
    WorkspaceController, 
    Agent, 
    AgentStatus,
    WorkspaceState,
    KeyboardHandler
)


class TestAgent:
    """Test Agent dataclass functionality"""
    
    def test_agent_creation(self):
        """Test creating a new agent"""
        agent = Agent(
            id="test-123",
            name="Test Agent", 
            command="test command",
            status=AgentStatus.SPAWNING,
            created_at=datetime.now()
        )
        
        assert agent.id == "test-123"
        assert agent.name == "Test Agent"
        assert agent.status == AgentStatus.SPAWNING
        assert agent.progress == 0.0
        assert len(agent.sequential_thoughts) == 0
        assert len(agent.output_lines) == 0
        
    def test_agent_add_thought(self):
        """Test adding sequential thinking steps"""
        agent = Agent(
            id="test",
            name="Test",
            command="test", 
            status=AgentStatus.THINKING,
            created_at=datetime.now()
        )
        
        agent.add_thought("First thought")
        agent.add_thought("Second thought")
        
        assert len(agent.sequential_thoughts) == 2
        assert agent.sequential_thoughts[0]['thought'] == "First thought"
        assert agent.sequential_thoughts[1]['thought'] == "Second thought"
        
    def test_agent_add_output(self):
        """Test adding output lines"""
        agent = Agent(
            id="test",
            name="Test", 
            command="test",
            status=AgentStatus.EXECUTING,
            created_at=datetime.now()
        )
        
        agent.add_output("Output line 1")
        agent.add_output("Output line 2")
        
        assert len(agent.output_lines) == 2
        assert agent.output_lines[0]['line'] == "Output line 1"
        assert agent.output_lines[1]['line'] == "Output line 2"
        
    def test_agent_thought_deque_limit(self):
        """Test that thoughts deque has max limit"""
        agent = Agent(
            id="test",
            name="Test",
            command="test",
            status=AgentStatus.THINKING, 
            created_at=datetime.now()
        )
        
        # Add more than 10 thoughts (the maxlen)
        for i in range(15):
            agent.add_thought(f"Thought {i}")
            
        # Should only keep last 10
        assert len(agent.sequential_thoughts) == 10
        assert agent.sequential_thoughts[0]['thought'] == "Thought 5"
        assert agent.sequential_thoughts[9]['thought'] == "Thought 14"


class TestWorkspaceState:
    """Test WorkspaceState functionality"""
    
    def test_workspace_state_creation(self):
        """Test creating workspace state"""
        state = WorkspaceState()
        
        assert len(state.agents) == 0
        assert state.selected_agent_id is None
        assert state.view_mode == "overview"
        assert state.total_agents_spawned == 0
        assert state.active_agents == 0
        
    def test_workspace_state_with_agents(self):
        """Test workspace state with agents"""
        state = WorkspaceState()
        
        agent1 = Agent("1", "Agent1", "cmd1", AgentStatus.EXECUTING, datetime.now())
        agent2 = Agent("2", "Agent2", "cmd2", AgentStatus.COMPLETED, datetime.now())
        
        state.agents["1"] = agent1
        state.agents["2"] = agent2
        state.selected_agent_id = "1"
        state.total_agents_spawned = 2
        
        assert len(state.agents) == 2
        assert state.selected_agent_id == "1"
        assert state.total_agents_spawned == 2


class TestKeyboardHandler:
    """Test KeyboardHandler functionality"""
    
    def test_keyboard_handler_creation(self):
        """Test creating keyboard handler"""
        handler = KeyboardHandler()
        
        assert len(handler.key_buffer) == 0
        assert not handler._running
        assert handler._thread is None
        
    def test_keyboard_handler_get_key_empty(self):
        """Test getting key from empty buffer"""
        handler = KeyboardHandler()
        
        key = handler.get_key()
        assert key is None
        
    def test_keyboard_handler_key_buffer(self):
        """Test key buffer functionality"""
        handler = KeyboardHandler()
        
        # Manually add keys to buffer
        handler.key_buffer.append('a')
        handler.key_buffer.append('b')
        
        assert handler.get_key() == 'a'
        assert handler.get_key() == 'b'
        assert handler.get_key() is None
        
    def test_keyboard_handler_buffer_limit(self):
        """Test key buffer respects maxlen"""
        handler = KeyboardHandler()
        
        # Fill beyond maxlen (100)
        for i in range(150):
            handler.key_buffer.append(str(i))
            
        assert len(handler.key_buffer) == 100
        # Should contain the last 100 keys
        assert handler.key_buffer[0] == "50"  # First key after overflow
        assert handler.key_buffer[-1] == "149"  # Last key added


class TestWorkspaceController:
    """Test WorkspaceController functionality"""
    
    def test_controller_creation(self):
        """Test creating workspace controller"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            
            assert controller.workspace_state is not None
            assert controller.keyboard_handler is not None
            assert controller.layout is not None
            assert not controller._running
            
    def test_controller_layout_setup(self):
        """Test layout structure is correct"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            
            # Check main layout sections exist
            assert "header" in controller.layout
            assert "main" in controller.layout
            assert "footer" in controller.layout
            assert "agents_panel" in controller.layout
            assert "detail_panel" in controller.layout
            assert "thinking_panel" in controller.layout
            assert "output_panel" in controller.layout
            
    def test_spawn_new_agent(self):
        """Test spawning a new agent"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            
            initial_count = len(controller.workspace_state.agents)
            controller._spawn_new_agent()
            
            assert len(controller.workspace_state.agents) == initial_count + 1
            assert controller.workspace_state.total_agents_spawned == 1
            assert controller.workspace_state.selected_agent_id is not None
            
            # Check agent properties
            agent_id = controller.workspace_state.selected_agent_id
            agent = controller.workspace_state.agents[agent_id]
            assert agent.status == AgentStatus.SPAWNING
            assert agent.name.startswith("Agent-")
            
    def test_kill_selected_agent(self):
        """Test killing selected agent"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            
            # Spawn an agent first
            controller._spawn_new_agent()
            agent_id = controller.workspace_state.selected_agent_id
            
            # Kill the agent
            controller._kill_selected_agent()
            
            agent = controller.workspace_state.agents[agent_id]
            assert agent.status == AgentStatus.KILLED
            assert agent.completed_at is not None
            
    def test_pause_resume_agent(self):
        """Test pausing and resuming agent"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            
            # Spawn an agent and set it executing
            controller._spawn_new_agent()
            agent_id = controller.workspace_state.selected_agent_id
            agent = controller.workspace_state.agents[agent_id]
            agent.status = AgentStatus.EXECUTING
            
            # Pause agent
            controller._pause_selected_agent()
            assert agent.status == AgentStatus.PAUSED
            
            # Resume agent
            controller._resume_selected_agent()
            assert agent.status == AgentStatus.EXECUTING
            
    def test_keyboard_input_handling(self):
        """Test keyboard input processing"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            controller.keyboard_handler.key_buffer.append('n')
            
            # Mock the get_key method to return our test key
            with patch.object(controller.keyboard_handler, 'get_key', return_value='n'):
                controller._handle_keyboard_input()
                
            # Should have spawned an agent
            assert len(controller.workspace_state.agents) == 1
            
    def test_keyboard_quit_handling(self):
        """Test quit command handling"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            controller._running = True
            
            with patch.object(controller.keyboard_handler, 'get_key', return_value='q'):
                controller._handle_keyboard_input()
                
            assert not controller._running
            
    def test_clear_agents_handling(self):
        """Test clear all agents command"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            
            # Add some agents
            controller._spawn_new_agent()
            controller._spawn_new_agent()
            
            assert len(controller.workspace_state.agents) == 2
            
            # Clear agents
            with patch.object(controller.keyboard_handler, 'get_key', return_value='c'):
                controller._handle_keyboard_input()
                
            assert len(controller.workspace_state.agents) == 0
            assert controller.workspace_state.selected_agent_id is None
            
    @patch('threading.Thread')
    def test_demo_agent_behavior(self, mock_thread):
        """Test demo agent behavior simulation"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            
            # Create a test agent
            agent_id = "test-123"
            agent = Agent(
                id=agent_id,
                name="Test Agent",
                command="test",
                status=AgentStatus.SPAWNING,
                created_at=datetime.now()
            )
            controller.workspace_state.agents[agent_id] = agent
            
            # Mock time.sleep to speed up the test
            with patch('time.sleep'):
                # Run demo behavior in foreground for testing
                controller._demo_agent_behavior(agent_id)
                
            # Agent should have progressed through states
            assert agent.status == AgentStatus.COMPLETED
            assert agent.progress == 1.0
            assert len(agent.sequential_thoughts) > 0
            assert len(agent.output_lines) > 0
            
    def test_active_agents_count(self):
        """Test active agents counting"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            
            # Add agents in different states
            agent1 = Agent("1", "Agent1", "cmd", AgentStatus.EXECUTING, datetime.now())
            agent2 = Agent("2", "Agent2", "cmd", AgentStatus.COMPLETED, datetime.now())
            agent3 = Agent("3", "Agent3", "cmd", AgentStatus.THINKING, datetime.now())
            agent4 = Agent("4", "Agent4", "cmd", AgentStatus.KILLED, datetime.now())
            
            controller.workspace_state.agents["1"] = agent1
            controller.workspace_state.agents["2"] = agent2  
            controller.workspace_state.agents["3"] = agent3
            controller.workspace_state.agents["4"] = agent4
            
            # Simulate the active agents counting logic
            active_count = sum(
                1 for agent in controller.workspace_state.agents.values()
                if agent.status not in [AgentStatus.COMPLETED, AgentStatus.ERROR, AgentStatus.KILLED]
            )
            
            assert active_count == 2  # agent1 (EXECUTING) and agent3 (THINKING)


class TestRenderingMethods:
    """Test rendering methods work without errors"""
    
    def test_render_header(self):
        """Test header rendering doesn't crash"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            
            panel = controller._render_header()
            assert panel is not None
            
    def test_render_agents_panel_empty(self):
        """Test agents panel with no agents"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            
            panel = controller._render_agents_panel()
            assert panel is not None
            
    def test_render_agents_panel_with_agents(self):
        """Test agents panel with agents"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            
            # Add test agent
            agent = Agent("1", "Test", "cmd", AgentStatus.EXECUTING, datetime.now())
            agent.progress = 0.5
            agent.current_task = "Testing"
            controller.workspace_state.agents["1"] = agent
            
            panel = controller._render_agents_panel()
            assert panel is not None
            
    def test_render_thinking_panel_no_selection(self):
        """Test thinking panel with no agent selected"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            
            panel = controller._render_thinking_panel()
            assert panel is not None
            
    def test_render_thinking_panel_with_agent(self):
        """Test thinking panel with selected agent"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            
            # Add agent with thoughts
            agent = Agent("1", "Test", "cmd", AgentStatus.THINKING, datetime.now())
            agent.add_thought("Test thought 1")
            agent.add_thought("Test thought 2")
            
            controller.workspace_state.agents["1"] = agent
            controller.workspace_state.selected_agent_id = "1"
            
            panel = controller._render_thinking_panel()
            assert panel is not None
            
    def test_render_output_panel_with_agent(self):
        """Test output panel with selected agent"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            
            # Add agent with output
            agent = Agent("1", "Test", "cmd", AgentStatus.EXECUTING, datetime.now())
            agent.add_output("Output line 1")
            agent.add_output("Output line 2")
            
            controller.workspace_state.agents["1"] = agent
            controller.workspace_state.selected_agent_id = "1"
            
            panel = controller._render_output_panel()
            assert panel is not None
            
    def test_render_footer(self):
        """Test footer rendering"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'):
            controller = WorkspaceController()
            
            panel = controller._render_footer()
            assert panel is not None


class TestPersistentBehavior:
    """Test persistent behavior like htop/k9s"""
    
    @pytest.mark.asyncio
    async def test_persistent_event_loop_setup(self):
        """Test that event loop is set up to run persistently"""
        with patch('src.agentsmcp.ui.v4.workspace_controller.Console'), \
             patch('src.agentsmcp.ui.v4.workspace_controller.Live') as mock_live:
            
            controller = WorkspaceController()
            
            # Mock the live context manager
            mock_live_context = MagicMock()
            mock_live.return_value.__enter__ = MagicMock(return_value=mock_live_context)
            mock_live.return_value.__exit__ = MagicMock(return_value=None)
            
            # Mock keyboard handler
            controller.keyboard_handler.start = MagicMock()
            controller.keyboard_handler.stop = MagicMock()
            
            # Mock _handle_keyboard_input to set _running to False after one iteration
            call_count = 0
            def mock_handle_input():
                nonlocal call_count
                call_count += 1
                if call_count >= 3:  # Stop after a few iterations
                    controller._running = False
                    
            controller._handle_keyboard_input = MagicMock(side_effect=mock_handle_input)
            controller._render_frame = MagicMock()
            
            # Run the controller
            await controller.run()
            
            # Verify persistent behavior setup
            controller.keyboard_handler.start.assert_called_once()
            controller.keyboard_handler.stop.assert_called_once()
            mock_live.assert_called_once()
            
            # Verify the loop ran multiple times
            assert controller._handle_keyboard_input.call_count >= 3
            assert controller._render_frame.call_count >= 1


# Integration test for the full controller
class TestIntegration:
    """Integration tests for full controller functionality"""
    
    @patch('src.agentsmcp.ui.v4.workspace_controller.Console')
    def test_full_agent_lifecycle(self, mock_console):
        """Test complete agent lifecycle management"""
        controller = WorkspaceController()
        
        # 1. Spawn agent
        controller._spawn_new_agent()
        assert len(controller.workspace_state.agents) == 1
        
        agent_id = controller.workspace_state.selected_agent_id
        agent = controller.workspace_state.agents[agent_id]
        
        # 2. Test agent progression
        agent.status = AgentStatus.THINKING
        agent.add_thought("Planning the task")
        
        agent.status = AgentStatus.EXECUTING  
        agent.add_output("Executing step 1")
        agent.progress = 0.5
        
        # 3. Test pause/resume
        controller._pause_selected_agent()
        assert agent.status == AgentStatus.PAUSED
        
        controller._resume_selected_agent()
        assert agent.status == AgentStatus.EXECUTING
        
        # 4. Test completion
        agent.status = AgentStatus.COMPLETED
        agent.progress = 1.0
        
        # 5. Verify state
        assert agent.progress == 1.0
        assert len(agent.sequential_thoughts) > 0
        assert len(agent.output_lines) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
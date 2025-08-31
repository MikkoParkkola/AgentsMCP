"""Lazy loading utilities and patterns for AgentsMCP system.

This module provides decorators, utilities and base classes to implement
lazy loading patterns throughout the system for improved startup performance.
"""

import functools
import importlib
import logging
import threading
import time
import weakref
from typing import Any, Callable, Dict, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LazyModule:
    """Lazy module loader that imports modules only when first accessed."""
    
    def __init__(self, module_name: str, package: Optional[str] = None):
        self._module_name = module_name
        self._package = package
        self._module: Optional[Any] = None
        self._import_lock = threading.Lock()
    
    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            with self._import_lock:
                if self._module is None:
                    try:
                        start_time = time.perf_counter()
                        self._module = importlib.import_module(self._module_name, self._package)
                        load_time = time.perf_counter() - start_time
                        logger.debug(f"Lazy loaded module {self._module_name} in {load_time:.3f}s")
                    except ImportError as e:
                        logger.error(f"Failed to lazy load module {self._module_name}: {e}")
                        raise
        
        return getattr(self._module, name)
    
    def __dir__(self) -> list[str]:
        if self._module is None:
            with self._import_lock:
                if self._module is None:
                    self._module = importlib.import_module(self._module_name, self._package)
        return dir(self._module)


class LazyProperty:
    """Descriptor for lazy property initialization."""
    
    def __init__(self, func: Callable[[Any], T]):
        self.func = func
        self.name = func.__name__
        self.__doc__ = func.__doc__
    
    def __get__(self, instance: Any, owner: Optional[type] = None) -> T:
        if instance is None:
            return self  # type: ignore
        
        # Use a private attribute name to store the cached value
        cache_attr = f'_lazy_{self.name}'
        
        if not hasattr(instance, cache_attr):
            start_time = time.perf_counter()
            value = self.func(instance)
            load_time = time.perf_counter() - start_time
            logger.debug(f"Lazy initialized {self.name} in {load_time:.3f}s")
            setattr(instance, cache_attr, value)
        
        return getattr(instance, cache_attr)
    
    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name


def lazy_import(module_name: str, package: Optional[str] = None) -> LazyModule:
    """Create a lazy module import.
    
    Args:
        module_name: Name of the module to import
        package: Package name for relative imports
        
    Returns:
        LazyModule instance that imports on first access
    """
    return LazyModule(module_name, package)


def memoized_property(func: Callable[[Any], T]) -> property:
    """Create a memoized property that caches the result after first computation.
    
    Args:
        func: Function to memoize as a property
        
    Returns:
        Property that computes value once and caches it
    """
    attr_name = f'_memoized_{func.__name__}'
    
    @functools.wraps(func)
    def wrapper(self: Any) -> T:
        if not hasattr(self, attr_name):
            start_time = time.perf_counter()
            value = func(self)
            compute_time = time.perf_counter() - start_time
            logger.debug(f"Memoized property {func.__name__} computed in {compute_time:.3f}s")
            setattr(self, attr_name, value)
        return getattr(self, attr_name)
    
    return property(wrapper)


class LazyFactory:
    """Factory for lazy instantiation of expensive objects."""
    
    def __init__(self, factory_func: Callable[..., T], *args: Any, **kwargs: Any):
        self._factory_func = factory_func
        self._args = args
        self._kwargs = kwargs
        self._instance: Optional[T] = None
        self._create_lock = threading.Lock()
    
    def get(self) -> T:
        """Get the instance, creating it lazily if needed."""
        if self._instance is None:
            with self._create_lock:
                if self._instance is None:
                    start_time = time.perf_counter()
                    self._instance = self._factory_func(*self._args, **self._kwargs)
                    create_time = time.perf_counter() - start_time
                    logger.debug(f"Lazy created {self._factory_func.__name__} instance in {create_time:.3f}s")
        return self._instance
    
    def is_created(self) -> bool:
        """Check if the instance has been created."""
        return self._instance is not None
    
    def reset(self) -> None:
        """Reset the factory, forcing recreation on next get()."""
        with self._create_lock:
            self._instance = None


class LazyRegistry:
    """Registry for lazy-loaded components with optional cleanup."""
    
    def __init__(self):
        self._factories: Dict[str, LazyFactory] = {}
        self._cleanup_funcs: Dict[str, Callable[[Any], None]] = {}
        self._registry_lock = threading.RLock()
    
    def register(self, 
                name: str, 
                factory_func: Callable[..., T], 
                *args: Any,
                cleanup_func: Optional[Callable[[T], None]] = None,
                **kwargs: Any) -> None:
        """Register a lazy factory for a component.
        
        Args:
            name: Component name
            factory_func: Function to create the component
            *args: Arguments for factory function
            cleanup_func: Optional cleanup function
            **kwargs: Keyword arguments for factory function
        """
        with self._registry_lock:
            if name in self._factories:
                logger.warning(f"Overriding existing lazy component: {name}")
            
            self._factories[name] = LazyFactory(factory_func, *args, **kwargs)
            if cleanup_func:
                self._cleanup_funcs[name] = cleanup_func
    
    def get(self, name: str) -> T:
        """Get a component, creating it lazily if needed.
        
        Args:
            name: Component name
            
        Returns:
            Component instance
            
        Raises:
            KeyError: If component is not registered
        """
        with self._registry_lock:
            if name not in self._factories:
                raise KeyError(f"Component not registered: {name}")
            
            return self._factories[name].get()
    
    def is_loaded(self, name: str) -> bool:
        """Check if a component has been loaded.
        
        Args:
            name: Component name
            
        Returns:
            True if component has been created
        """
        with self._registry_lock:
            if name not in self._factories:
                return False
            return self._factories[name].is_created()
    
    def reset(self, name: str) -> None:
        """Reset a component, forcing recreation on next access.
        
        Args:
            name: Component name
        """
        with self._registry_lock:
            if name in self._factories:
                # Run cleanup if available
                if name in self._cleanup_funcs and self._factories[name].is_created():
                    try:
                        instance = self._factories[name].get()
                        self._cleanup_funcs[name](instance)
                    except Exception as e:
                        logger.warning(f"Cleanup failed for {name}: {e}")
                
                self._factories[name].reset()
    
    def cleanup_all(self) -> None:
        """Run cleanup for all created components."""
        with self._registry_lock:
            for name in list(self._factories.keys()):
                if self._factories[name].is_created():
                    self.reset(name)
    
    def list_components(self) -> Dict[str, bool]:
        """List all registered components and their loaded status.
        
        Returns:
            Dict mapping component names to loaded status
        """
        with self._registry_lock:
            return {name: factory.is_created() for name, factory in self._factories.items()}


def lazy_cached(maxsize: int = 128) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for lazy caching with size limit.
    
    Args:
        maxsize: Maximum cache size
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: Dict[tuple, T] = {}
        cache_lock = threading.RLock()
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create cache key from args and kwargs
            key = (args, tuple(sorted(kwargs.items())))
            
            with cache_lock:
                if key in cache:
                    return cache[key]
                
                # If cache is full, remove oldest entry
                if len(cache) >= maxsize:
                    # Remove first item (oldest in insertion order)
                    first_key = next(iter(cache))
                    del cache[first_key]
                
                # Compute and cache result
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                compute_time = time.perf_counter() - start_time
                logger.debug(f"Cached {func.__name__} computed in {compute_time:.3f}s")
                
                cache[key] = result
                return result
        
        # Add cache inspection methods
        wrapper.cache_info = lambda: {"size": len(cache), "maxsize": maxsize}  # type: ignore
        wrapper.cache_clear = lambda: cache.clear()  # type: ignore
        
        return wrapper
    
    return decorator


# Global lazy registry instance
_global_registry = LazyRegistry()

def register_lazy_component(name: str, 
                           factory_func: Callable[..., T], 
                           *args: Any,
                           cleanup_func: Optional[Callable[[T], None]] = None,
                           **kwargs: Any) -> None:
    """Register a component in the global lazy registry."""
    _global_registry.register(name, factory_func, *args, cleanup_func=cleanup_func, **kwargs)


def get_lazy_component(name: str) -> Any:
    """Get a component from the global lazy registry."""
    return _global_registry.get(name)


def is_component_loaded(name: str) -> bool:
    """Check if a component is loaded in the global registry."""
    return _global_registry.is_loaded(name)


def reset_lazy_component(name: str) -> None:
    """Reset a component in the global registry."""
    _global_registry.reset(name)


def cleanup_all_lazy_components() -> None:
    """Cleanup all components in the global registry."""
    _global_registry.cleanup_all()


def get_lazy_registry_status() -> Dict[str, bool]:
    """Get the status of all lazy components."""
    return _global_registry.list_components()


def get_performance_monitor():
    """Get the performance monitor instance.
    
    This is a convenience function that lazy-imports the performance module
    and returns the global performance monitor instance.
    """
    from .performance import get_performance_monitor as _get_performance_monitor
    return _get_performance_monitor()
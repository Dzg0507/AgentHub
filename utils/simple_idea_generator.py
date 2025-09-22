#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple AI-Powered Project Idea Generator
Generates quick project ideas using the AI agents
"""
import random
from typing import List

class SimpleIdeaGenerator:
    """Simple AI-powered project idea generator"""
    
    def __init__(self):
        # Simple predefined ideas for quick generation
        self.quick_ideas = [
            "Create a task management app with drag-and-drop functionality",
            "Build a weather dashboard with real-time data visualization",
            "Make a password generator with customizable options",
            "Develop a file organizer that sorts files by type and date",
            "Create a simple calculator with advanced mathematical functions",
            "Build a todo list app with categories and priorities",
            "Make a random quote generator with different categories",
            "Develop a basic text editor with syntax highlighting",
            "Create a number guessing game with multiple difficulty levels",
            "Build a simple chat application with real-time messaging",
            "Make a expense tracker with category-based budgeting",
            "Develop a note-taking app with search and tagging",
            "Create a basic drawing application with different tools",
            "Build a simple music player with playlist management",
            "Make a basic web scraper for collecting data from websites",
            "Develop a simple blog platform with user authentication",
            "Create a basic e-commerce store with shopping cart",
            "Build a simple social media dashboard",
            "Make a basic game like tic-tac-toe or snake",
            "Develop a simple API for data management"
        ]
    
    def get_quick_idea(self) -> str:
        """Get a random quick idea"""
        return random.choice(self.quick_ideas)
    
    def get_multiple_ideas(self, count: int = 4) -> List[str]:
        """Get multiple random ideas"""
        return random.sample(self.quick_ideas, min(count, len(self.quick_ideas)))
    
    async def generate_ai_idea(self, agents: dict, category: str = "general") -> str:
        """Generate an idea using AI agents"""
        try:
            # Use the prompt enhancer agent to generate a creative project idea
            idea_response = agents["prompt_enhancer"].generate_reply(
                messages=[{
                    "role": "user", 
                    "content": f"Generate a creative and practical project idea for a {category} application. Keep it simple but interesting. Just give me the project description in one sentence."
                }]
            )
            
            idea_content = idea_response.get("content", "") if isinstance(idea_response, dict) else str(idea_response)
            
            # Extract just the idea description
            if idea_content.strip():
                return idea_content.strip()
            else:
                # Fallback to predefined idea
                return self.get_quick_idea()
                
        except Exception as e:
            # Fallback to predefined idea if AI fails
            return self.get_quick_idea()

# Global simple generator instance
simple_idea_generator = SimpleIdeaGenerator()

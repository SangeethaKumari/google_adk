#!/usr/bin/env python3
"""
Simple script to retrieve and print all voice-to-text messages and images.
Can be run independently or used as a reference for API calls.
"""

import requests
import json

API_BASE = "http://localhost:8100"

def get_all_messages():
    """Get all voice-to-text messages from all conversations."""
    print("Fetching all voice-to-text messages...")
    response = requests.get(f"{API_BASE}/api/all-messages")
    
    if response.status_code == 200:
        data = response.json()
        print("\n" + "="*80)
        print("ALL VOICE-TO-TEXT MESSAGES")
        print("="*80)
        print(f"Total Sessions: {data['total_sessions']}")
        print(f"Total User Messages: {data['total_user_messages']}")
        print("\nCombined Text:")
        print("-"*80)
        print(data['combined_text'])
        print("="*80)
        return data
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def get_all_images():
    """Get all images saved in the directory."""
    print("\nFetching all images...")
    response = requests.get(f"{API_BASE}/api/all-images")
    
    if response.status_code == 200:
        data = response.json()
        print("\n" + "="*80)
        print("ALL IMAGES")
        print("="*80)
        print(f"Total Images: {data['total_images']}")
        print("-"*80)
        for img in data['images']:
            print(f"Session: {img['session_id']}")
            print(f"Filename: {img['filename']}")
            print(f"Path: {img['file_path']}")
            print(f"Size: {img['size_kb']} KB")
            print("-"*80)
        print("="*80)
        return data
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

if __name__ == "__main__":
    print("Patient Advocacy - Message & Image Retrieval")
    print("="*80)
    
    # Get all messages
    messages = get_all_messages()
    
    # Get all images
    images = get_all_images()
    
    # Save to file if data retrieved
    if messages:
        with open("all_messages.json", "w") as f:
            json.dump(messages, f, indent=2)
        print("\n✅ Saved all messages to 'all_messages.json'")
    
    if images:
        with open("all_images.json", "w") as f:
            json.dump(images, f, indent=2)
        print("✅ Saved all images info to 'all_images.json'")


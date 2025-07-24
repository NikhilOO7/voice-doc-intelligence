# apps/api/services/voice/__init__.py
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from livekit import api
from typing import Dict, Any, Optional
import logging

from ...core.config import settings
from ...core.connections import connections

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/create-room")
async def create_voice_room(
    room_name: Optional[str] = None,
    participant_name: str = "User"
) -> Dict[str, Any]:
    """Create a LiveKit room for voice interaction"""
    
    try:
        # Generate room name if not provided
        if not room_name:
            import uuid
            room_name = f"doc-intelligence-{uuid.uuid4().hex[:8]}"
        
        # Create LiveKit room
        room_service = api.RoomService()
        room_service.api_key = settings.livekit_api_key
        room_service.api_secret = settings.livekit_api_secret
        room_service.url = settings.livekit_url
        
        # Create or get room
        room = await room_service.create_room(
            api.CreateRoomRequest(
                name=room_name,
                empty_timeout=300,  # 5 minutes
                max_participants=5,
            )
        )
        
        # Generate access token for participant
        token = api.AccessToken(
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret
        )
        
        token.with_identity(participant_name)
        token.with_name(participant_name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        ))
        
        jwt_token = token.to_jwt()
        
        return {
            "room_name": room_name,
            "token": jwt_token,
            "url": settings.livekit_url,
            "participant_name": participant_name,
            "room_info": {
                "name": room.name,
                "creation_time": room.creation_time,
                "max_participants": room.max_participants,
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to create voice room: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create voice room: {str(e)}")

@router.post("/join-room")
async def join_voice_room(
    room_name: str,
    participant_name: str = "User"
) -> Dict[str, Any]:
    """Generate token to join existing voice room"""
    
    try:
        # Generate access token
        token = api.AccessToken(
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret
        )
        
        token.with_identity(participant_name)
        token.with_name(participant_name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        ))
        
        jwt_token = token.to_jwt()
        
        return {
            "room_name": room_name,
            "token": jwt_token,
            "url": settings.livekit_url,
            "participant_name": participant_name
        }
        
    except Exception as e:
        logger.error(f"Failed to join voice room: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to join room: {str(e)}")

@router.get("/rooms")
async def list_active_rooms() -> Dict[str, Any]:
    """List active voice rooms"""
    
    try:
        room_service = api.RoomService()
        room_service.api_key = settings.livekit_api_key
        room_service.api_secret = settings.livekit_api_secret
        room_service.url = settings.livekit_url
        
        rooms = await room_service.list_rooms(api.ListRoomsRequest())
        
        room_list = []
        for room in rooms.rooms:
            room_list.append({
                "name": room.name,
                "creation_time": room.creation_time,
                "num_participants": room.num_participants,
                "max_participants": room.max_participants,
                "metadata": room.metadata,
            })
        
        return {
            "rooms": room_list,
            "total": len(room_list)
        }
        
    except Exception as e:
        logger.error(f"Failed to list rooms: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list rooms: {str(e)}")

@router.delete("/room/{room_name}")
async def delete_voice_room(room_name: str) -> Dict[str, Any]:
    """Delete a voice room"""
    
    try:
        room_service = api.RoomService()
        room_service.api_key = settings.livekit_api_key
        room_service.api_secret = settings.livekit_api_secret
        room_service.url = settings.livekit_url
        
        await room_service.delete_room(api.DeleteRoomRequest(room=room_name))
        
        return {
            "message": f"Room '{room_name}' deleted successfully",
            "room_name": room_name
        }
        
    except Exception as e:
        logger.error(f"Failed to delete room: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete room: {str(e)}")

@router.get("/health")
async def voice_health_check() -> Dict[str, Any]:
    """Health check for voice services"""
    
    health_status = {
        "status": "healthy",
        "livekit_configured": bool(settings.livekit_api_key and settings.livekit_api_secret),
        "openai_configured": bool(settings.openai_api_key),
    }
    
    # Check if we can connect to LiveKit
    try:
        room_service = api.RoomService()
        room_service.api_key = settings.livekit_api_key
        room_service.api_secret = settings.livekit_api_secret
        room_service.url = settings.livekit_url
        
        # Test connection by listing rooms
        await room_service.list_rooms(api.ListRoomsRequest())
        health_status["livekit_connection"] = "healthy"
        
    except Exception as e:
        health_status["livekit_connection"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status
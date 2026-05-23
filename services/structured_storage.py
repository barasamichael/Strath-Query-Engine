"""
Structured Data Storage Service for KnowStrath RAG System

This service handles storage and querying of structured schedule data alongside
the existing vector database, providing SQL-like query capabilities.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import openai
import os

logger = logging.getLogger("structured_storage")


class StructuredDataStorage:
    """Manages structured storage for schedule data with SQL query capabilities."""

    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize the structured storage.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize OpenAI client for text-to-SQL conversion
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Create database and tables
        self._initialize_database()

    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create schedules table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schedules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    class_group TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    unit_code TEXT,
                    room TEXT NOT NULL,
                    day TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    instructor TEXT,
                    session_type TEXT,
                    semester TEXT NOT NULL,
                    raw_content TEXT,
                    confidence REAL DEFAULT 1.0,
                    extraction_timestamp TEXT,
                    doc_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indexes for better query performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_class_group ON schedules(class_group)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_day ON schedules(day)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_room ON schedules(room)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_time ON schedules(day, start_time, end_time)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_subject ON schedules(subject)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_instructor ON schedules(instructor)"
            )

            # Create document metadata table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schedule_documents (
                    doc_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    total_entries INTEGER DEFAULT 0,
                    extraction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    file_hash TEXT
                )
            """
            )

            conn.commit()
            logger.info("Structured database initialized successfully")

    def store_schedules(
        self,
        schedules: List[Dict[str, Any]],
        doc_id: str,
        file_name: str,
        file_path: str,
    ) -> int:
        """
        Store schedule entries in the database.

        Args:
            schedules: List of schedule entry dictionaries
            doc_id: Document identifier
            file_name: Original file name
            file_path: Original file path

        Returns:
            Number of entries stored
        """
        if not schedules:
            return 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            try:
                # Store schedule entries
                for schedule in schedules:
                    cursor.execute(
                        """
                        INSERT INTO schedules (
                            class_group, subject, unit_code, room, day, 
                            start_time, end_time, instructor, session_type,
                            semester, raw_content, confidence, extraction_timestamp, doc_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            schedule.get("class_group"),
                            schedule.get("subject"),
                            schedule.get("unit_code"),
                            schedule.get("room"),
                            schedule.get("day"),
                            schedule.get("start_time"),
                            schedule.get("end_time"),
                            schedule.get("instructor"),
                            schedule.get("session_type"),
                            schedule.get("semester"),
                            schedule.get("raw_content"),
                            schedule.get("confidence", 1.0),
                            schedule.get(
                                "extraction_timestamp",
                                datetime.now().isoformat(),
                            ),
                            doc_id,
                        ),
                    )

                # Store document metadata
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO schedule_documents 
                    (doc_id, file_name, file_path, total_entries)
                    VALUES (?, ?, ?, ?)
                """,
                    (doc_id, file_name, file_path, len(schedules)),
                )

                conn.commit()

                print(
                    f"📊 Stored {len(schedules)} schedule entries for {file_name}"
                )
                logger.info(
                    f"Stored {len(schedules)} schedule entries from {file_name}"
                )
                return len(schedules)

            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to store schedules: {e}")
                raise

    def query_with_sql(self, sql_query: str) -> List[Dict[str, Any]]:
        """
        Execute a SQL query against the schedules table.

        Args:
            sql_query: SQL query string

        Returns:
            List of result dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable dict-like access
                cursor = conn.cursor()

                cursor.execute(sql_query)
                results = [dict(row) for row in cursor.fetchall()]

                logger.info(f"SQL query returned {len(results)} results")
                return results

        except Exception as e:
            logger.error(f"SQL query failed: {e}")
            return []

    def query_with_natural_language(self, query: str) -> Dict[str, Any]:
        """
        Convert natural language query to SQL and execute it.

        Args:
            query: Natural language query about schedules

        Returns:
            Dictionary with SQL query, results, and formatted response
        """
        try:
            # Convert to SQL
            sql_query = self._convert_to_sql(query)

            if not sql_query:
                return {
                    "success": False,
                    "error": "Could not generate SQL query",
                    "results": [],
                }

            # Execute SQL
            results = self.query_with_sql(sql_query)

            # Format response
            formatted_response = self._format_query_response(query, results)

            return {
                "success": True,
                "sql_query": sql_query,
                "results": results,
                "formatted_response": formatted_response,
                "result_count": len(results),
            }

        except Exception as e:
            logger.error(f"Natural language query failed: {e}")
            return {"success": False, "error": str(e), "results": []}

    def _convert_to_sql(self, natural_query: str) -> Optional[str]:
        """Convert natural language query to SQL using LLM."""

        schema_info = """
        Table: schedules
        Columns:
        - class_group (TEXT): Class identifier like "BICS 1A"
        - subject (TEXT): Subject name like "Object Oriented Programming"
        - unit_code (TEXT): Course code like "ICS1202"
        - room (TEXT): Room/location like "LT 3", "Lab 2"
        - day (TEXT): Day of week like "Monday", "Tuesday"
        - start_time (TEXT): Start time in HH:MM format like "08:15"
        - end_time (TEXT): End time in HH:MM format like "10:15"
        - instructor (TEXT): Instructor name like "Dr. Nelson Ochieng"
        - session_type (TEXT): Type like "lecture", "lab", "practical"
        - semester (TEXT): Semester like "August-December 2025"
        """

        prompt = f"""
        Convert this natural language question to a SQL query for the schedules table.
        
        Schema:
        {schema_info}
        
        Question: "{natural_query}"
        
        Example conversions:
        "What classes does BICS 1A have on Monday?" 
        → SELECT * FROM schedules WHERE class_group = 'BICS 1A' AND day = 'Monday'
        
        "Which rooms are free at 2 PM on Tuesday?"
        → SELECT DISTINCT room FROM schedules WHERE room NOT IN (
            SELECT room FROM schedules WHERE day = 'Tuesday' AND start_time <= '14:00' AND end_time > '14:00'
        )
        
        "Who teaches Database Systems?"
        → SELECT DISTINCT instructor FROM schedules WHERE subject LIKE '%Database%'
        
        "What time is Object Oriented Programming for BICS 1A?"
        → SELECT day, start_time, end_time, room FROM schedules WHERE class_group = 'BICS 1A' AND subject LIKE '%Object Oriented%'
        
        Return only the SQL query, no other text.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )

            sql_query = response.choices[0].message.content.strip()

            # Clean up the response (remove any extra text)
            if sql_query.lower().startswith("select"):
                return sql_query
            else:
                # Try to extract SQL from response
                lines = sql_query.split("\n")
                for line in lines:
                    if line.strip().lower().startswith("select"):
                        return line.strip()

            return None

        except Exception as e:
            logger.error(f"SQL conversion failed: {e}")
            return None

    def _format_query_response(
        self, original_query: str, results: List[Dict[str, Any]]
    ) -> str:
        """Format query results into a natural language response."""

        if not results:
            return "I couldn't find any matching schedule information for your query."

        # Limit results for response formatting
        limited_results = results[:10]  # Show first 10 results

        prompt = f"""
        Format these database results into a natural, helpful response for the user.
        
        User asked: "{original_query}"
        
        Database results: {json.dumps(limited_results, indent=2)}
        
        Guidelines:
        - Provide a clear, natural language answer
        - Include relevant details like times, rooms, instructors
        - If there are many results, summarize appropriately
        - Be concise but informative
        - If results are schedule entries, format times nicely (e.g., "8:15 AM to 10:15 AM")
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400,
            )

            formatted_response = response.choices[0].message.content.strip()

            # Add result count if there were more results
            if len(results) > 10:
                formatted_response += (
                    f"\n\n(Showing first 10 of {len(results)} results)"
                )

            return formatted_response

        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            # Fallback to basic formatting
            return self._basic_format_results(original_query, limited_results)

    def _basic_format_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> str:
        """Basic fallback formatting for query results."""

        if not results:
            return "No matching schedule information found."

        response_lines = []

        for result in results[:5]:  # Show first 5
            if "subject" in result and "day" in result:
                line = f"{result.get('class_group', '')} has {result.get('subject', 'Unknown')} on {result.get('day', 'Unknown')}"
                if result.get("start_time") and result.get("end_time"):
                    line += (
                        f" from {result['start_time']} to {result['end_time']}"
                    )
                if result.get("room"):
                    line += f" in {result['room']}"
                response_lines.append(line)
            else:
                # Generic formatting for other types of results
                line = ", ".join(
                    f"{k}: {v}" for k, v in result.items() if v is not None
                )
                response_lines.append(line)

        response = "\n".join(response_lines)

        if len(results) > 5:
            response += f"\n\n(Showing first 5 of {len(results)} results)"

        return response

    def get_available_rooms(
        self, day: str, start_time: str, end_time: str
    ) -> List[str]:
        """Get rooms that are available during a specific time slot."""

        sql_query = """
        SELECT DISTINCT room 
        FROM schedules 
        WHERE room NOT IN (
            SELECT room 
            FROM schedules 
            WHERE day = ? 
            AND NOT (end_time <= ? OR start_time >= ?)
        )
        ORDER BY room
        """

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql_query, (day, start_time, end_time))
            return [row[0] for row in cursor.fetchall()]

    def get_class_schedule(
        self, class_group: str, day: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get schedule for a specific class group."""

        base_query = "SELECT * FROM schedules WHERE class_group = ?"
        params = [class_group]

        if day:
            base_query += " AND day = ?"
            params.append(day)

        base_query += " ORDER BY day, start_time"

        return self.query_with_sql(base_query)

    def get_instructor_schedule(self, instructor: str) -> List[Dict[str, Any]]:
        """Get all classes taught by a specific instructor."""

        sql_query = """
        SELECT * FROM schedules 
        WHERE instructor LIKE ? 
        ORDER BY day, start_time
        """

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql_query, (f"%{instructor}%",))
            return [dict(row) for row in cursor.fetchall()]

    def get_room_schedule(self, room: str) -> List[Dict[str, Any]]:
        """Get all classes scheduled in a specific room."""

        sql_query = """
        SELECT * FROM schedules 
        WHERE room LIKE ? 
        ORDER BY day, start_time
        """

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql_query, (f"%{room}%",))
            return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            stats = {}

            # Total entries
            cursor.execute("SELECT COUNT(*) FROM schedules")
            stats["total_entries"] = cursor.fetchone()[0]

            # Unique class groups
            cursor.execute("SELECT COUNT(DISTINCT class_group) FROM schedules")
            stats["unique_class_groups"] = cursor.fetchone()[0]

            # Unique subjects
            cursor.execute("SELECT COUNT(DISTINCT subject) FROM schedules")
            stats["unique_subjects"] = cursor.fetchone()[0]

            # Unique rooms
            cursor.execute("SELECT COUNT(DISTINCT room) FROM schedules")
            stats["unique_rooms"] = cursor.fetchone()[0]

            # Unique instructors
            cursor.execute("SELECT COUNT(DISTINCT instructor) FROM schedules")
            stats["unique_instructors"] = cursor.fetchone()[0]

            # Documents processed
            cursor.execute("SELECT COUNT(*) FROM schedule_documents")
            stats["documents_processed"] = cursor.fetchone()[0]

            return stats

    def delete_document_schedules(self, doc_id: str) -> int:
        """Delete all schedule entries for a specific document."""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Count entries to be deleted
            cursor.execute(
                "SELECT COUNT(*) FROM schedules WHERE doc_id = ?", (doc_id,)
            )
            count = cursor.fetchone()[0]

            # Delete entries
            cursor.execute("DELETE FROM schedules WHERE doc_id = ?", (doc_id,))
            cursor.execute(
                "DELETE FROM schedule_documents WHERE doc_id = ?", (doc_id,)
            )

            conn.commit()

            logger.info(
                f"Deleted {count} schedule entries for document {doc_id}"
            )
            return count

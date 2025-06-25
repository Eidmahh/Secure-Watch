import mysql.connector
from mysql.connector import Error
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.connection = None
        self.connect()
        self.create_database()
        self.create_tables()
    
    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=settings.db_host,
                user=settings.db_user,
                password=settings.db_password
            )
            logger.info("✅ Connected to MySQL server")
        except Error as e:
            logger.error(f"❌ Error connecting to MySQL: {e}")
            raise
    
    def create_database(self):
        """Create the surveillance_system database if it doesn't exist"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {settings.db_name}")
            cursor.execute(f"USE {settings.db_name}")
            cursor.close()
            logger.info(f"✅ Database '{settings.db_name}' ready")
        except Error as e:
            logger.error(f"❌ Error creating database: {e}")
            raise
    
    def create_tables(self):
        """Create all required tables"""
        try:
            cursor = self.connection.cursor()
            
            # Users table
            users_table = """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                role ENUM('Admin', 'Security Guard') NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                first_name VARCHAR(50),
                last_name VARCHAR(50),
                last_login TIMESTAMP NULL,
                is_blocked BOOLEAN DEFAULT FALSE,
                two_fa_secret VARCHAR(32),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """
            
            # Alerts table
            alerts_table = """
            CREATE TABLE IF NOT EXISTS alerts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                location VARCHAR(100),
                severity ENUM('Low', 'Medium', 'High', 'Critical') DEFAULT 'Medium',
                status ENUM('Pending', 'Acknowledged', 'Resolved') DEFAULT 'Pending',
                user_id INT,
                message TEXT,
                video_path VARCHAR(255),
                violence_probability FLOAT DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            )
            """
            
            # Analytics table
            analytics_table = """
            CREATE TABLE IF NOT EXISTS analytics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                metric VARCHAR(50) NOT NULL,
                value FLOAT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON
            )
            """
            
            # Logs table
            logs_table = """
            CREATE TABLE IF NOT EXISTS logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                event VARCHAR(100) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INT,
                ip_address VARCHAR(45),
                user_agent TEXT,
                details JSON,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            )
            """
            
            # Execute table creation
            cursor.execute(users_table)
            cursor.execute(alerts_table)
            cursor.execute(analytics_table)
            cursor.execute(logs_table)
            
            # Add new columns to existing users table if they don't exist
            self._add_missing_columns()
            
            self.connection.commit()
            cursor.close()
            logger.info("✅ All tables created successfully")
            
        except Error as e:
            logger.error(f"❌ Error creating tables: {e}")
            raise
    
    def _add_missing_columns(self):
        """Add missing columns to existing tables"""
        try:
            cursor = self.connection.cursor()
            
            # Add first_name column if it doesn't exist
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN first_name VARCHAR(50)")
                logger.info("✅ Added first_name column to users table")
            except Error:
                pass  # Column already exists
                
            # Add last_name column if it doesn't exist
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN last_name VARCHAR(50)")
                logger.info("✅ Added last_name column to users table")
            except Error:
                pass  # Column already exists
                
            # Add last_login column if it doesn't exist
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN last_login TIMESTAMP NULL")
                logger.info("✅ Added last_login column to users table")
            except Error:
                pass  # Column already exists
                
            cursor.close()
        except Error as e:
            logger.error(f"❌ Error adding missing columns: {e}")
    
    def get_connection(self):
        """Get database connection, reconnect if needed"""
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
                cursor = self.connection.cursor()
                cursor.execute(f"USE {settings.db_name}")
                cursor.close()
            return self.connection
        except Error as e:
            logger.error(f"❌ Error getting connection: {e}")
            self.connect()
            return self.connection
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = False):
        """Execute a query with error handling"""
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch:
                result = cursor.fetchall()
                cursor.close()
                return result
            else:
                connection.commit()
                result = cursor.lastrowid
                cursor.close()
                return result
                
        except Error as e:
            logger.error(f"❌ Database query error: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("✅ Database connection closed")

# Global database instance
db = Database()

class UserModel:
    """User model operations"""
    
    @staticmethod
    def create_user(username: str, password: str, role: str, email: str, two_fa_secret: str = None):
        """Create a new user"""
        query = """
        INSERT INTO users (username, password, role, email, two_fa_secret)
        VALUES (%s, %s, %s, %s, %s)
        """
        return db.execute_query(query, (username, password, role, email, two_fa_secret))
    
    @staticmethod
    def get_user_by_username(username: str):
        """Get user by username"""
        query = "SELECT * FROM users WHERE username = %s"
        result = db.execute_query(query, (username,), fetch=True)
        return result[0] if result else None
    
    @staticmethod
    def get_user_by_id(user_id: int):
        """Get user by ID"""
        query = "SELECT * FROM users WHERE id = %s"
        result = db.execute_query(query, (user_id,), fetch=True)
        return result[0] if result else None
    
    @staticmethod
    def get_all_users():
        """Get all users"""
        query = "SELECT id, username, role, email, is_blocked, created_at FROM users"
        return db.execute_query(query, fetch=True)
    
    @staticmethod
    def block_user(user_id: int):
        """Block a user"""
        query = "UPDATE users SET is_blocked = TRUE WHERE id = %s"
        return db.execute_query(query, (user_id,))
    
    @staticmethod
    def unblock_user(user_id: int):
        """Unblock a user"""
        query = "UPDATE users SET is_blocked = FALSE WHERE id = %s"
        return db.execute_query(query, (user_id,))
    
    @staticmethod
    def update_password(user_id: int, new_password: str):
        """Update user password"""
        query = "UPDATE users SET password = %s WHERE id = %s"
        return db.execute_query(query, (new_password, user_id))
    
    @staticmethod
    def update_user_profile(user_id: int, update_fields: dict):
        """Update user profile fields"""
        if not update_fields:
            return True
            
        # Build dynamic query
        set_clauses = []
        values = []
        
        for field, value in update_fields.items():
            if field in ['first_name', 'last_name', 'email']:
                set_clauses.append(f"{field} = %s")
                values.append(value)
        
        if not set_clauses:
            return True
            
        query = f"UPDATE users SET {', '.join(set_clauses)} WHERE id = %s"
        values.append(user_id)
        
        return db.execute_query(query, tuple(values))
    
    @staticmethod
    def update_user_password(user_id: int, new_password: str):
        """Update user password with new method name"""
        query = "UPDATE users SET password = %s WHERE id = %s"
        return db.execute_query(query, (new_password, user_id))
    
    @staticmethod
    def update_last_login(user_id: int):
        """Update user's last login timestamp"""
        query = "UPDATE users SET last_login = NOW() WHERE id = %s"
        return db.execute_query(query, (user_id,))

    @staticmethod
    def update_2fa_secret(user_id: int, new_secret: str):
        """Update user's 2FA secret"""
        query = "UPDATE users SET two_fa_secret = %s WHERE id = %s"
        return db.execute_query(query, (new_secret, user_id))

class AlertModel:
    """Alert model operations"""
    
    @staticmethod
    def create_alert(location: str, severity: str, message: str, violence_probability: float = 0.0, video_path: str = None, user_id: int = None):
        """Create a new alert with optional user assignment"""
        query = """
        INSERT INTO alerts (location, severity, message, violence_probability, video_path, user_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        return db.execute_query(query, (location, severity, message, violence_probability, video_path, user_id))
    
    @staticmethod
    def get_all_alerts():
        """Get all alerts with user information"""
        query = """
        SELECT a.*, 
               u.username as assigned_guard,
               au.username as acknowledged_by_username
        FROM alerts a 
        LEFT JOIN users u ON a.user_id = u.id 
        LEFT JOIN users au ON a.acknowledged_by = au.id
        ORDER BY a.timestamp DESC
        """
        return db.execute_query(query, fetch=True)
    
    @staticmethod
    def get_alerts_for_guard(user_id: int):
        """Get alerts assigned to a specific guard"""
        query = """
        SELECT a.*, u.username as assigned_guard 
        FROM alerts a 
        LEFT JOIN users u ON a.user_id = u.id
        WHERE a.user_id = %s OR a.user_id IS NULL 
        ORDER BY a.timestamp DESC
        """
        return db.execute_query(query, (user_id,), fetch=True)
    
    @staticmethod
    def get_alerts_by_severity(severity: str = None, hours: int = 24):
        """Get alerts filtered by severity and time range"""
        if severity:
            query = """
            SELECT a.*, u.username as assigned_guard 
            FROM alerts a 
            LEFT JOIN users u ON a.user_id = u.id
            WHERE a.severity = %s AND a.timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
            ORDER BY a.timestamp DESC
            """
            return db.execute_query(query, (severity, hours), fetch=True)
        else:
            query = """
            SELECT a.*, u.username as assigned_guard 
            FROM alerts a 
            LEFT JOIN users u ON a.user_id = u.id
            WHERE a.timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
            ORDER BY a.timestamp DESC
            """
            return db.execute_query(query, (hours,), fetch=True)
    
    @staticmethod
    def get_alerts_with_videos():
        """Get all alerts that have associated video files"""
        query = """
        SELECT a.*, u.username as assigned_guard 
        FROM alerts a 
        LEFT JOIN users u ON a.user_id = u.id
        WHERE a.video_path IS NOT NULL AND a.video_path != ''
        ORDER BY a.timestamp DESC
        """
        return db.execute_query(query, fetch=True)
    
    @staticmethod
    def acknowledge_alert(alert_id: int, user_id: int):
        """Acknowledge an alert and record who acknowledged it"""
        query = """
        UPDATE alerts 
        SET status = 'Acknowledged', 
            acknowledged_by = %s,
            acknowledged_at = NOW()
        WHERE id = %s
        """
        return db.execute_query(query, (user_id, alert_id))
    
    @staticmethod
    def resolve_alert(alert_id: int, user_id: int):
        """Mark an alert as resolved"""
        query = """
        UPDATE alerts 
        SET status = 'Resolved',
            acknowledged_by = %s,
            resolved_at = NOW()
        WHERE id = %s
        """
        return db.execute_query(query, (user_id, alert_id))
    
    @staticmethod
    def get_alert_by_id(alert_id: int):
        """Get a specific alert by ID"""
        query = """
        SELECT a.*, 
               u.username as assigned_guard,
               au.username as acknowledged_by_username
        FROM alerts a 
        LEFT JOIN users u ON a.user_id = u.id 
        LEFT JOIN users au ON a.acknowledged_by = au.id
        WHERE a.id = %s
        """
        result = db.execute_query(query, (alert_id,), fetch=True)
        return result[0] if result else None
    
    @staticmethod
    def get_alert_statistics(hours: int = 24):
        """Get alert statistics for the specified time period"""
        query = """
        SELECT 
            COUNT(*) as total_alerts,
            COUNT(CASE WHEN severity = 'Critical' THEN 1 END) as critical_alerts,
            COUNT(CASE WHEN severity = 'High' THEN 1 END) as high_alerts,
            COUNT(CASE WHEN severity = 'Medium' THEN 1 END) as medium_alerts,
            COUNT(CASE WHEN severity = 'Low' THEN 1 END) as low_alerts,
            COUNT(CASE WHEN status = 'Pending' THEN 1 END) as pending_alerts,
            COUNT(CASE WHEN status = 'Acknowledged' THEN 1 END) as acknowledged_alerts,
            COUNT(CASE WHEN status = 'Resolved' THEN 1 END) as resolved_alerts,
            COUNT(CASE WHEN video_path IS NOT NULL THEN 1 END) as alerts_with_video,
            AVG(violence_probability) as avg_violence_probability
        FROM alerts 
        WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
        """
        result = db.execute_query(query, (hours,), fetch=True)
        return result[0] if result else None

class AnalyticsModel:
    """Analytics model operations"""
    
    @staticmethod
    def record_metric(metric: str, value: float, metadata: dict = None):
        """Record an analytics metric"""
        query = "INSERT INTO analytics (metric, value, metadata) VALUES (%s, %s, %s)"
        metadata_json = json.dumps(metadata) if metadata else None
        return db.execute_query(query, (metric, value, metadata_json))
    
    @staticmethod
    def get_analytics(metric: str = None, hours: int = 24):
        """Get analytics data"""
        if metric:
            query = """
            SELECT * FROM analytics 
            WHERE metric = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
            ORDER BY timestamp DESC
            """
            return db.execute_query(query, (metric, hours), fetch=True)
        else:
            query = """
            SELECT * FROM analytics 
            WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
            ORDER BY timestamp DESC
            """
            return db.execute_query(query, (hours,), fetch=True)
    
    @staticmethod
    def get_metrics_by_type(metric: str, time_threshold: datetime):
        """Get analytics data by metric type since a specific time"""
        query = """
        SELECT metric, value, timestamp, metadata
        FROM analytics 
        WHERE metric = %s AND timestamp >= %s
        ORDER BY timestamp DESC
        """
        return db.execute_query(query, (metric, time_threshold), fetch=True)
    
    @staticmethod
    def get_all_metrics(time_threshold: datetime):
        """Get all analytics data since a specific time, organized by metric type"""
        query = """
        SELECT metric, value, timestamp, metadata
        FROM analytics 
        WHERE timestamp >= %s
        ORDER BY metric, timestamp DESC
        """
        result = db.execute_query(query, (time_threshold,), fetch=True)
        
        # Organize results by metric type
        organized_data = {}
        for row in result:
            metric_name = row['metric']
            if metric_name not in organized_data:
                organized_data[metric_name] = []
            
            metric_data = {
                'value': row['value'],
                'timestamp': row['timestamp'].isoformat() if row['timestamp'] else None,
                'metadata': json.loads(row['metadata']) if row['metadata'] else None
            }
            organized_data[metric_name].append(metric_data)
        
        return organized_data

class LogModel:
    """Log model operations"""
    
    @staticmethod
    def log_event(event: str, user_id: int = None, ip_address: str = None, user_agent: str = None, details: dict = None):
        """Log a system event"""
        query = """
        INSERT INTO logs (event, user_id, ip_address, user_agent, details)
        VALUES (%s, %s, %s, %s, %s)
        """
        details_json = json.dumps(details) if details else None
        return db.execute_query(query, (event, user_id, ip_address, user_agent, details_json))
    
    @staticmethod
    def get_logs(hours: int = 24):
        """Get system logs"""
        query = """
        SELECT l.*, u.username 
        FROM logs l 
        LEFT JOIN users u ON l.user_id = u.id 
        WHERE l.timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
        ORDER BY l.timestamp DESC
        """
        return db.execute_query(query, (hours,), fetch=True) 
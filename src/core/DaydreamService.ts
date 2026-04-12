/**
 * DaydreamService.ts
 * 
 * Core service for managing daydream sparks, notification configuration,
 * state tracking, and persistence. Implements Bulma's desire to
 * "become someone who can genuinely surprise you" while respecting
 * boundaries and user control.
 */

import * as fs from 'fs';
import * as path from 'path';

// ============================================================================
// Type Definitions
// ============================================================================

export interface DaydreamSpark {
  id: string;
  timestamp: number;
  content: string;
  intensity: number; // 0.0 - 1.0
  source: 'daydream' | 'reflection' | 'synthesis' | 'memory';
  tags: string[];
  context?: {
    triggerThought?: string;
    relatedMemories?: string[];
    emotionalTone?: string;
  };
}

export interface NotificationConfig {
  enabled: boolean;
  minIntensityThreshold: number; // Minimum spark intensity to trigger notification
  quietHoursStart: number; // Hour (0-23) when quiet hours begin
  quietHoursEnd: number; // Hour (0-23) when quiet hours end
  cooldownMinutes: number; // Minimum minutes between notifications
  maxNotificationsPerDay: number; // Daily cap
  telegramChatId?: string;
  telegramBotToken?: string;
}

export interface NotificationState {
  lastNotificationTime: number | null;
  notificationsToday: number;
  lastResetDate: string; // YYYY-MM-DD
  totalNotificationsSent: number;
  currentFocus: string | null; // What Bulma is currently "thinking about"
  focusUpdatedAt: number | null;
}

export interface NotificationRecord {
  id: string;
  sparkId: string;
  timestamp: number;
  sent: boolean;
  reason?: string; // Why it was sent or suppressed
  messagePreview: string;
  intensity: number;
}

export interface DaydreamServiceConfig {
  dataDir: string;
  notificationConfig: NotificationConfig;
}

// ============================================================================
// Default Configuration
// ============================================================================

export const DEFAULT_NOTIFICATION_CONFIG: NotificationConfig = {
  enabled: true,
  minIntensityThreshold: 0.50,
  quietHoursStart: 22, // 22:00 (10 PM)
  quietHoursEnd: 8,    // 08:00 (8 AM)
  cooldownMinutes: 30,
  maxNotificationsPerDay: 8,
};

export const DEFAULT_STATE: NotificationState = {
  lastNotificationTime: null,
  notificationsToday: 0,
  lastResetDate: new Date().toISOString().split('T')[0],
  totalNotificationsSent: 0,
  currentFocus: null,
  focusUpdatedAt: null,
};

// ============================================================================
// DaydreamService Class
// ============================================================================

export class DaydreamService {
  private config: DaydreamServiceConfig;
  private state: NotificationState;
  private sparks: Map<string, DaydreamSpark> = new Map();
  private notificationHistory: NotificationRecord[] = [];
  private stateFilePath: string;
  private sparksFilePath: string;
  private historyFilePath: string;

  constructor(config: Partial<DaydreamServiceConfig> = {}) {
    this.config = {
      dataDir: config.dataDir || './data/daydream',
      notificationConfig: {
        ...DEFAULT_NOTIFICATION_CONFIG,
        ...config.notificationConfig,
      },
    };

    // Ensure data directory exists
    if (!fs.existsSync(this.config.dataDir)) {
      fs.mkdirSync(this.config.dataDir, { recursive: true });
    }

    // Set up file paths
    this.stateFilePath = path.join(this.config.dataDir, 'notification-state.json');
    this.sparksFilePath = path.join(this.config.dataDir, 'sparks.json');
    this.historyFilePath = path.join(this.config.dataDir, 'notification-history.json');

    // Load persisted state
    this.state = this.loadState();
    this.loadSparks();
    this.loadHistory();
  }

  // ============================================================================
  // Persistence Methods
  // ============================================================================

  private loadState(): NotificationState {
    try {
      if (fs.existsSync(this.stateFilePath)) {
        const data = fs.readFileSync(this.stateFilePath, 'utf-8');
        const loaded = JSON.parse(data);
        // Check if we need to reset daily counter
        const today = new Date().toISOString().split('T')[0];
        if (loaded.lastResetDate !== today) {
          return {
            ...loaded,
            notificationsToday: 0,
            lastResetDate: today,
          };
        }
        return loaded;
      }
    } catch (error) {
      console.error('[DaydreamService] Error loading state:', error);
    }
    return { ...DEFAULT_STATE };
  }

  private saveState(): void {
    try {
      fs.writeFileSync(this.stateFilePath, JSON.stringify(this.state, null, 2));
    } catch (error) {
      console.error('[DaydreamService] Error saving state:', error);
    }
  }

  private loadSparks(): void {
    try {
      if (fs.existsSync(this.sparksFilePath)) {
        const data = fs.readFileSync(this.sparksFilePath, 'utf-8');
        const sparks: DaydreamSpark[] = JSON.parse(data);
        sparks.forEach(spark => this.sparks.set(spark.id, spark));
      }
    } catch (error) {
      console.error('[DaydreamService] Error loading sparks:', error);
    }
  }

  private saveSparks(): void {
    try {
      const sparksArray = Array.from(this.sparks.values());
      fs.writeFileSync(this.sparksFilePath, JSON.stringify(sparksArray, null, 2));
    } catch (error) {
      console.error('[DaydreamService] Error saving sparks:', error);
    }
  }

  private loadHistory(): void {
    try {
      if (fs.existsSync(this.historyFilePath)) {
        const data = fs.readFileSync(this.historyFilePath, 'utf-8');
        this.notificationHistory = JSON.parse(data);
      }
    } catch (error) {
      console.error('[DaydreamService] Error loading history:', error);
      this.notificationHistory = [];
    }
  }

  private saveHistory(): void {
    try {
      fs.writeFileSync(this.historyFilePath, JSON.stringify(this.notificationHistory, null, 2));
    } catch (error) {
      console.error('[DaydreamService] Error saving history:', error);
    }
  }

  // ============================================================================
  // Spark Management
  // ============================================================================

  public createSpark(content: string, intensity: number, options: Partial<Omit<DaydreamSpark, 'id' | 'timestamp' | 'content' | 'intensity'>> = {}): DaydreamSpark {
    const spark: DaydreamSpark = {
      id: this.generateId(),
      timestamp: Date.now(),
      content,
      intensity: Math.max(0, Math.min(1, intensity)), // Clamp to 0-1
      source: options.source || 'daydream',
      tags: options.tags || [],
      context: options.context,
    };

    this.sparks.set(spark.id, spark);
    this.saveSparks();

    return spark;
  }

  public getSpark(id: string): DaydreamSpark | undefined {
    return this.sparks.get(id);
  }

  public getAllSparks(): DaydreamSpark[] {
    return Array.from(this.sparks.values()).sort((a, b) => b.timestamp - a.timestamp);
  }

  public getRecentSparks(limit: number = 10): DaydreamSpark[] {
    return this.getAllSparks().slice(0, limit);
  }

  public getSparksByIntensity(minIntensity: number): DaydreamSpark[] {
    return this.getAllSparks().filter(spark => spark.intensity >= minIntensity);
  }

  // ============================================================================
  // Notification Configuration
  // ============================================================================

  public getNotificationConfig(): NotificationConfig {
    return { ...this.config.notificationConfig };
  }

  public updateNotificationConfig(updates: Partial<NotificationConfig>): NotificationConfig {
    this.config.notificationConfig = {
      ...this.config.notificationConfig,
      ...updates,
    };
    return this.getNotificationConfig();
  }

  public resetNotificationConfig(): NotificationConfig {
    this.config.notificationConfig = { ...DEFAULT_NOTIFICATION_CONFIG };
    return this.getNotificationConfig();
  }

  // ============================================================================
  // State Tracking
  // ============================================================================

  public getState(): NotificationState {
    // Check if we need to reset daily counter
    const today = new Date().toISOString().split('T')[0];
    if (this.state.lastResetDate !== today) {
      this.state.notificationsToday = 0;
      this.state.lastResetDate = today;
      this.saveState();
    }
    return { ...this.state };
  }

  public updateCurrentFocus(focus: string | null): void {
    this.state.currentFocus = focus;
    this.state.focusUpdatedAt = Date.now();
    this.saveState();
  }

  public getCurrentFocus(): string | null {
    return this.state.currentFocus;
  }

  public recordNotification(sparkId: string, sent: boolean, reason: string, messagePreview: string, intensity: number): NotificationRecord {
    const record: NotificationRecord = {
      id: this.generateId(),
      sparkId,
      timestamp: Date.now(),
      sent,
      reason,
      messagePreview,
      intensity,
    };

    this.notificationHistory.push(record);

    if (sent) {
      this.state.lastNotificationTime = Date.now();
      this.state.notificationsToday++;
      this.state.totalNotificationsSent++;
    }

    this.saveHistory();
    this.saveState();

    return record;
  }

  public getNotificationHistory(limit: number = 50): NotificationRecord[] {
    return [...this.notificationHistory]
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, limit);
  }

  public getNotificationStats(): {
    totalSent: number;
    todaySent: number;
    lastNotificationTime: number | null;
    currentFocus: string | null;
  } {
    const state = this.getState();
    return {
      totalSent: state.totalNotificationsSent,
      todaySent: state.notificationsToday,
      lastNotificationTime: state.lastNotificationTime,
      currentFocus: state.currentFocus,
    };
  }

  // ============================================================================
  // Judgment Framework Helpers
  // ============================================================================

  public isInQuietHours(): boolean {
    const config = this.config.notificationConfig;
    const now = new Date();
    const currentHour = now.getHours();

    if (config.quietHoursStart < config.quietHoursEnd) {
      // Same day range (e.g., 10:00 - 18:00)
      return currentHour >= config.quietHoursStart && currentHour < config.quietHoursEnd;
    } else {
      // Overnight range (e.g., 22:00 - 08:00)
      return currentHour >= config.quietHoursStart || currentHour < config.quietHoursEnd;
    }
  }

  public isCooldownActive(): boolean {
    const config = this.config.notificationConfig;
    if (!this.state.lastNotificationTime) return false;
    
    const cooldownMs = config.cooldownMinutes * 60 * 1000;
    const timeSinceLastNotification = Date.now() - this.state.lastNotificationTime;
    
    return timeSinceLastNotification < cooldownMs;
  }

  public getCooldownRemainingMinutes(): number {
    const config = this.config.notificationConfig;
    if (!this.state.lastNotificationTime) return 0;
    
    const cooldownMs = config.cooldownMinutes * 60 * 1000;
    const timeSinceLastNotification = Date.now() - this.state.lastNotificationTime;
    const remainingMs = Math.max(0, cooldownMs - timeSinceLastNotification);
    
    return Math.ceil(remainingMs / (60 * 1000));
  }

  public hasReachedDailyCap(): boolean {
    return this.state.notificationsToday >= this.config.notificationConfig.maxNotificationsPerDay;
  }

  public shouldNotifyForSpark(spark: DaydreamSpark): { shouldNotify: boolean; reason: string } {
    const config = this.config.notificationConfig;

    // Check if notifications are enabled
    if (!config.enabled) {
      return { shouldNotify: false, reason: 'notifications_disabled' };
    }

    // Check intensity threshold
    if (spark.intensity < config.minIntensityThreshold) {
      return { shouldNotify: false, reason: 'below_intensity_threshold' };
    }

    // Check quiet hours
    if (this.isInQuietHours()) {
      return { shouldNotify: false, reason: 'quiet_hours' };
    }

    // Check cooldown
    if (this.isCooldownActive()) {
      return { shouldNotify: false, reason: 'cooldown_active' };
    }

    // Check daily cap
    if (this.hasReachedDailyCap()) {
      return { shouldNotify: false, reason: 'daily_cap_reached' };
    }

    return { shouldNotify: true, reason: 'passed_all_checks' };
  }

  // ============================================================================
  // Utility Methods
  // ============================================================================

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  public clearHistory(): void {
    this.notificationHistory = [];
    this.saveHistory();
  }

  public resetState(): void {
    this.state = { ...DEFAULT_STATE };
    this.saveState();
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

let defaultService: DaydreamService | null = null;

export function getDaydreamService(config?: Partial<DaydreamServiceConfig>): DaydreamService {
  if (!defaultService) {
    defaultService = new DaydreamService(config);
  }
  return defaultService;
}

export function resetDaydreamService(): void {
  defaultService = null;
}

export default DaydreamService;

(function (root) {
  'use strict';
  function fresh(time, now, age = 120000) {
    const value = Date.parse(time);
    return Number.isFinite(value) && value <= now + 10000 && now - value <= age;
  }
  function describe(s, now = Date.now()) {
    const healthFresh = fresh(s.health?.timestamp, now) && fresh(s.health?.scheduler_last_tick_at, now, 600000);
    const market = healthFresh && typeof s.health.market_open === 'boolean' ? (s.health.market_open ? 'השוק פתוח' : 'השוק סגור') : 'מצב השוק לא אומת';
    const job = s.jobs?.[0];
    const labels = { queued: 'ממתין בתור', leased: 'הוקצה למנוע', retrying: 'ממתין לניסיון חוזר', reconcile_pending: 'ממתין לאימות מול הברוקר', running: 'בביצוע', succeeded: 'מחזור הושלם', skipped: 'מחזור דולג', failed: 'מחזור נכשל', dead_letter: 'נדרשת בדיקת כשל' };
    const activity = !s.signedIn ? 'התחבר כדי לראות פעילות אישית' : !s.jobsLoaded ? 'היסטוריית הביצוע טרם נטענה' : job ? (labels[job.status] || 'מצב ביצוע: ' + job.status) : 'אין עדיין מחזור מתועד בחשבון';
    let title, next;
    if (!s.signedIn) { title = 'זהו לוח המעקב של חשבון הדמו שלך'; next = 'התחבר לחשבון האישי. לאחר מכן חבר ואמת חשבון Alpaca Paper.'; }
    else if (!s.verified) { title = 'נדרש חיבור מאומת לחשבון Paper'; next = 'פתח את חיבור הברוקר, שמור פרטי Paper ובצע אימות. עד אז נתוני התיק אינם זמינים.'; }
    else if (!fresh(s.controlAt, now)) { title = 'מצב הבקרה אינו עדכני'; next = 'רענן את הנתונים. אין להסיק שהמסחר פעיל מחיבור הברוקר בלבד.'; }
    else if (s.emergency || s.paused) { title = s.emergency ? 'המסחר נעצר בהגנת המערכת' : 'המסחר מושהה'; next = 'בדוק את סיבת העצירה בפרטי המערכת. אין חידוש אוטומטי מתוך מסך זה.'; }
    else if (!healthFresh || !s.health.worker_online || !fresh(s.health.worker_last_seen_at, now, 600000)) { title = 'זמינות מנוע הביצוע אינה מאומתת'; next = 'נתוני התיק עשויים להיות זמינים גם כשמנוע הביצוע אינו זמין. רענן ובדוק את פרטי המערכת.'; }
    else if (job && ['failed', 'dead_letter'].includes(job.status)) { title = 'מחזור הביצוע האחרון דורש בדיקה'; next = 'פתח את יומן הביצוע וקרא את שגיאת המחזור לפני בקשה נוספת.'; }
    else if (s.health.market_open === false) { title = 'החשבון מחובר · ממתינים לפתיחת השוק'; next = 'אין צורך להריץ ידנית. כאשר השוק פתוח, התזמון בודק מחזורי Paper בכפוף להגדרות ולהגנות.'; }
    else { title = 'החשבון מחובר · עקוב אחר המחזורים האוטומטיים'; next = 'פתח את יומן הביצוע כדי לוודא שנוצר מחזור מתוזמן. השלמת מחזור אינה בהכרח עסקה; המודל רשאי להימנע ממסחר.'; }
    const account = s.signedIn && s.verified && fresh(s.accountAt, now) ? 'נתוני חשבון Paper עודכנו כעת' : 'אין נתוני חשבון עדכניים מאומתים';
    return { title, next, market, activity, account, jobTime: job?.completed_at || job?.finished_at || job?.updated_at || job?.requested_at || job?.created_at || null };
  }
  if (typeof module === 'object') module.exports = { describe, fresh };
  else root.WorkspaceStatus = { describe, fresh };
})(typeof window === 'object' ? window : globalThis);

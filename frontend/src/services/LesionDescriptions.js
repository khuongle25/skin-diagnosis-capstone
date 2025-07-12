const lesionDescriptions = {
  MEL: {
    name: "Melanoma",
    description:
      "Melanoma là một dạng ung thư da ác tính bắt nguồn từ các tế bào sắc tố melanin. Đây là loại nguy hiểm nhất trong các loại ung thư da vì có khả năng di căn nhanh và đe dọa đến tính mạng nếu không được phát hiện sớm.",
    severity: "high",
    icon: "warning",
  },
  NV: {
    name: "Melanocytic Nevi",
    description:
      "Melanocytic Nevi hay còn gọi là nốt ruồi, là những tổn thương lành tính hình thành do sự tăng sinh của tế bào sắc tố. Đây là loại tổn thương phổ biến và thường không nguy hiểm.",
    severity: "low",
    icon: "check_circle",
  },
  BCC: {
    name: "Basal Cell Carcinoma",
    description:
      "Ung thư biểu mô tế bào đáy là loại ung thư da phổ biến nhất, khởi phát từ lớp đáy của biểu bì. BCC thường phát triển chậm và hiếm khi di căn, nhưng có thể gây tổn thương nghiêm trọng nếu không được điều trị.",
    severity: "medium",
    icon: "report_problem",
  },
  AK: {
    name: "Actinic Keratosis / Bowen's Disease",
    description:
      "AKIEC là thuật ngữ chung cho các tổn thương tiền ung thư da như keratosis quang hóa (actinic keratosis) và bệnh Bowen. Những tổn thương này có khả năng tiến triển thành ung thư tế bào vảy nếu không được theo dõi và xử lý kịp thời.",
    severity: "medium",
    icon: "report_problem",
  },
  BKL: {
    name: "Benign Keratosis-like Lesions",
    description:
      "BKL là nhóm tổn thương lành tính trên da, bao gồm các dạng như dày sừng tiết bã (seborrheic keratosis), dày sừng ánh sáng và các tổn thương tương tự. Chúng thường có bề ngoài sẫm màu, thô ráp nhưng không mang tính ác tính.",
    severity: "low",
    icon: "check_circle",
  },
  DF: {
    name: "Dermatofibroma",
    description:
      "Dermatofibroma là một loại u da lành tính, thường nhỏ, chắc và có màu nâu hoặc đỏ. Chúng thường xuất hiện ở chân hoặc cánh tay và không gây hại, tuy nhiên có thể bị nhầm lẫn với tổn thương ác tính.",
    severity: "low",
    icon: "check_circle",
  },
  VASC: {
    name: "Vascular Lesions",
    description:
      "Tổn thương mạch máu (vascular lesions) bao gồm các dị dạng hoặc tăng sinh mạch máu như u máu (hemangiomas), bớt rượu vang (port wine stains)... Chúng thường có màu đỏ hoặc tím và lành tính.",
    severity: "low",
    icon: "check_circle",
  },
  SCC: {
    name: "Squamous Cell Carcinoma",
    description:
      "Ung thư biểu mô tế bào vảy là một dạng ung thư da ác tính bắt nguồn từ tế bào vảy ở lớp ngoài của da. SCC có thể lan sang các cơ quan khác nếu không được điều trị kịp thời, tuy nhiên ít nguy hiểm hơn melanoma.",
    severity: "medium",
    icon: "report_problem",
  },
  UNK: {
    name: "Unknown",
    description:
      "Không có thông tin mô tả cụ thể cho nhãn này. Đây có thể là các trường hợp chưa được xác định rõ hoặc không thuộc bất kỳ nhãn nào còn lại.",
    severity: "unknown",
    icon: "help",
  },
};

const getLesionDescription = (lesionType) => {
  if (!lesionType) return null;

  // Tách lấy mã tổn thương (thường nằm trong ngoặc đơn)
  const match = lesionType.match(/\(([^)]+)\)/);
  const code = match ? match[1] : lesionType.split(" ")[0];

  return lesionDescriptions[code] || null;
};

const getCodeFromFullName = (fullName) => {
  const mapping = {
    "Melanoma": "MEL",
    "Melanocytic Nevi": "NV", 
    "Basal Cell Carcinoma": "BCC",
    "Actinic Keratosis / Bowen's Disease": "AK",
    "Benign Keratosis-like Lesions": "BKL",
    "Dermatofibroma": "DF",
    "Squamous Cell Carcinoma": "SCC",
    "Unknown": "UNK",
    "Vascular Lesions": "VASC"
  };
  return mapping[fullName] || fullName;
};

export { getLesionDescription as default, getCodeFromFullName }
